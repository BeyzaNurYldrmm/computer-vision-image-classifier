# ConvNeXT 

# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath



# define ConvNeXT main layer
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):# dim:kanal sayısı
        super().__init__()
        self.dwconv= nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm= LayerNorm(dim, eps=1e-6)
        self.pwconv1= nn.Linear(dim, 4*dim)
        self.act=nn.GELU()
        self.pwconv2= nn.Linear(4*dim, dim)
        self.gamma= nn.Parameter(layer_scale_init_value*torch.ones((dim)),
                                 requires_grad=True) if layer_scale_init_value >0 else None
        
        self.drop_path= DropPath(drop_path) if drop_path >0. else nn.Identity()

    def forward(self, x):
        input=x
        x=self.dwconv(x)
        x= x.permute(0,2,3,1) #(N, C, H, W)=>(N, H, W, C)
        x=self.norm(x)
        x= self.pwconv1(x)
        x=self.act(x)
        x= self.pwconv2(x)
        if self.gamma is not None:
            x= self.gamma*x

        x= x.permute(0,3,1,2) #(N, H,W,C)=>(N,C,H,W)
        x= input+ self.drop_path(x)
        return x

# define LayerNorm
    #-----------------------------------------------------------------------#
    # Custom Layernorm Default Channels_Last
    #    channels_last  [batch_size, height, width, channels]
    #    channels_first [batch_size, channels, height, width]
    #-------------------------------------------------------------------------#

class LayerNorm(nn.Module):#pytorcta  channel_last(N;C;H;W) versiyonu cnn'de channel_first (N;H;W;C)** iki versiyonuda destekleyen layerNorm
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # Gradient hesaplanır, optimizer ile güncellenir
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
        
        if self.data_format not in ["channels_last", "channels_first"]: 
            #hep channels_last ı kullanacagız pytorch old.
            raise NotImplementedError(" data_format desteklenmiyor.")

    def forward(self, x):  
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":  # bu kısmı kullanmayacagız bunu kapatıp adımsal verim alabiliriz
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
# Main Model
class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=100, 
                depths=[3,3,9,3], dims=[96,192,384,768], drop_path_rate=0.,
                layer_scale_init_value=1e-6, head_init_scale=1.):
        """
        in_chan Giriş kanal sayısı (RGB resimler için 3)
        num_classes Çıkış sınıf sayısı (ImageNet için 1000)
        depths Her stage'deki blok sayıları        
        dims Her stage'deki kanal sayıları        
        drop_path_rate Regularization için dropout benzeri teknik
        """
        super().__init__()

        # Save STEM and Sampling
        self.downsample_layers = nn.ModuleList()  # [batch_size,3,224,224] -> [batch_size,dim[0],56,56]
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # -> Pooling işlemini 2x2 boyutunda ve 2 adım (stride) ile bir konvolüsyon ile değiştir
        # Bu işlem tüm stage'lerde bir kez uygulanır.
        for i in range(3):  # her stemden sonra 1 downsample
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dims[i], drop_path=dp_rates[cur+j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # son layer norm
        self.head = nn.Linear(dims[-1], num_classes)  
    
        self.apply(self._init_weights)
        if head_init_scale > 0:
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):  # feature extraction (özellik çıkarıcı)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
    
            
    def forward(self, x):  # head (sınıflandırma)
        x = self.forward_features(x)
        x = self.head(x)
        return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

def register_model(func):
    return func

@register_model
def convnext_tiny(pretrained=True, num_classes=1000, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.head = torch.nn.Linear(768, num_classes)
        state_dict = checkpoint["model"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}
        model.load_state_dict(state_dict, strict=False)

    return model


@register_model
def convnext_small(pretrained=False, num_classes=1000, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
        model.head = torch.nn.Linear(768, num_classes)
    return model


def convnext_model(dims, cn_drop_path_rate,layer_scale_init_value, head_init_scale, depths,k, in_chans, num_class, device="cpu", ft=True):

    if ft:
        model = convnext_tiny(pretrained = ft, num_classes=num_class).to(device)

        for param in model.parameters():
            param.requires_grad = False
    
        model.head = nn.Sequential(nn.Linear(768, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 20))
    
    
        for param in model.head.parameters():
            param.requires_grad = True
    
        # Unfreeze the last stage
        for param in model.stages[k].parameters():
            param.requires_grad = True
            
    else:
        model= ConvNeXt(in_chans, num_class, depths, dims, cn_drop_path_rate,layer_scale_init_value, head_init_scale)        
    
    return model