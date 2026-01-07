#efficientnet mdl

import torch.nn as nn
from math import ceil
import torch
from torchsummary import summary
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

basic_mb_params = [
    # k, channels(c), repeats(t), stride(s), kernel_size(k)
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

alpha, beta = 1.2, 1.1

scale_values = {
    # (phi, resolution, dropout)
    "b0": (0, 224, 0.2),
    # "b0": (4, 224, 0.4),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}
# DepthwiseConv. Block define=> conv-norm-akt
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.cnnblock = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, groups=groups),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU())

    def forward(self, x):
        return self.cnnblock(x)
    
# MBCovn block define => Dar-Genişlet-Depthwise conv- sıkıştır-residual
class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride, padding, ratio, reduction=2,
    ):
        super(MBBlock, self).__init__()
        # self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * ratio # ratio=1 ise genisletmeye gerek yok darbogaz uygulama
        self.expand = in_channels != hidden_dim

        # This is for squeeze and excitation block
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim,
                kernel_size=3,stride=1,padding=1)

        self.conv = nn.Sequential(
            ConvBlock(hidden_dim if self.expand else in_channels,
                      hidden_dim if self.expand else in_channels,
                      kernel_size, stride, padding, 
                      groups=hidden_dim if self.expand else 1),
            SqueezeExcitation(hidden_dim if self.expand else in_channels, reduced_dim),
            nn.Conv2d(hidden_dim if self.expand else in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs):
        if self.expand:
          x = self.expand_conv(inputs)#depthwise
        else:
          x = inputs
        return self.conv(x)


#SE kanalsal dikkat mekanizmasını
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)
    
#final model
class EfficientNet(nn.Module):
    def __init__(self, model_name, output):
        super(EfficientNet, self).__init__()
        phi, resolution, dropout = scale_values[model_name] # phi olcek sbt
        self.depth_factor, self.width_factor = alpha**phi, beta**phi
        self.last_channels = ceil(1280 * self.width_factor)
        self.avgpool= nn.AdaptiveAvgPool2d(1)
        self.feature_extractor()# conv2d+mbconv
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channels, output),
        )

    def feature_extractor(self):
        channels = int(32 * self.width_factor)
        features = [ConvBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for k, c_o, repeat, s, n in basic_mb_params:
            #k=> genisletme ks, c_o=> cikis kanali, repeat=>kac kez tekrar edilecegi, s=>ilk mb blockda stride, 
            #n=>kernel boyut
            # For numeric stability, we multiply and divide by 4
            out_channels = 4 * ceil(int(c_o * self.width_factor) / 4)
            num_layers = ceil(repeat * self.depth_factor)

            for layer in range(num_layers):
                stride = s if layer == 0 else 1
                features.append(
                    MBBlock(in_channels, out_channels, kernel_size=n, stride=stride, padding=n//2, ratio=k)
                )
                in_channels = out_channels  # sonraki layer için out_channels geç

        features.append(ConvBlock(in_channels, self.last_channels, kernel_size=1, stride=1, padding=0))
        self.extractor = nn.Sequential(*features)

    def forward(self, x): 
        x = self.extractor(x)
        """ Görselle katmanları
        state_visu="feacture_maps"
        os.makedirs(state_visu, exist_ok=True)
        y=0
        for i, layer in enumerate(self.extractor):
            x = layer(x)
            
            if y == 0:
                y=1
                for c in channels_to_show:
                    fmap = x[0, c].detach().cpu()
                    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
                    plt.figure(figsize=(3,3))
                    plt.imshow(fmap, cmap="viridis")
                    plt.title(f"Block_{i}_{layer.__class__.__name__}")
                    plt.axis("off")
                    filename= f"Convnext_model_block_{i}_{layer.__class__.__name__}_channel_{c}.jpg"
                    filepath=os.path.join(state_visu, filename)
                    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, format="jpg")
                    plt.close()
              """
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.classifier(x)
    
    
def get_efficientnet(num_classes, device):
    model = efficientnet_b0(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                    num_classes)

    return model.to(device)    



    
def efficientnet(mdl_name, ft, num_class, device):
    if ft is False:
        if mdl_name == "efficientnet_b0":
            model = EfficientNet("b0", num_class).to(device)
            # summary(model, (3, 224, 224))
        
        """
        elif mdl_name == "efficientnet_b1":
            model = EfficientNet("b1", num_class).to(device)
            summary(model, (3, 240, 240))

        elif mdl_name == "efficientnet_b7":
            model = EfficientNet("b7", num_class).to(device)
            summary(model, (3, 600, 600))

        else:
            model = EfficientNet("b0", num_class).to(device)
            summary(model, (3, 224, 224))
            """

    elif ft == True:
        model = get_efficientnet(num_class, device)

    return model

       