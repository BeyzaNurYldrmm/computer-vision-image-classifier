# inference.py
import sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm

def load_image(image_path: Path, image_size=224):
    transform = T.Compose([
        T.Resize(image_size + 32),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def run_inference(
    model_name: str,
    weights_path: Path,
    image_path: Path,
    num_class: int,
    image_size: int = 224
):
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "ViT":
            model = timm.create_model("vit_large_r50_s32_224", pretrained=False, num_classes=num_class).to(device)
            
    elif model_name == "ConvNext":
        model = timm.create_model("convnext_tiny", pretrained =False, num_classes=num_class).to(device)

    elif model_name == "Swin":
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_class)

    elif model_name == "EfficientNet":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_class
        )
        model = model.to(device)
        
    elif model_name == "BEiT":
        model = timm.create_model("beitv2_base_patch16_224", pretrained=False, num_classes=num_class )

    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")
    

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    x = load_image(image_path, image_size).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        
    return model_name, pred.item(), conf.item() * 100