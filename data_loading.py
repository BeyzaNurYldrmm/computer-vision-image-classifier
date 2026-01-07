#data process
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from pathlib import Path

#data loading func
def get_data_loader(cfg):
    data_cfg= cfg["data"]

    train_path= Path(data_cfg["train_path"])
    test_path = Path(data_cfg["test_path"])
    batch_size= data_cfg["batch_size"]
    image_size= data_cfg.get("image_size", 224)
    
    transform_train=transforms.Compose([
        #transforms.Resize((224,224)),
       # transforms.ToTensor(),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])       
    ])
    
    transform_test= transforms.Compose([
       transforms.Resize(image_size+32),               
       transforms.CenterCrop(image_size),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                      std=[0.229, 0.224, 0.225])  
   ])
    
    train_set= datasets.ImageFolder(train_path, transform=transform_train)
    test_set= datasets.ImageFolder(test_path, transform=transform_test)
    
    val_size = len(test_set) // 2
    test_size = len(test_set) - val_size
    
    val_dataset, test_dataset = random_split(test_set, [val_size, test_size])

    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
        "val":   DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }

    return loaders, len(train_set.classes)

#data visualization
def visualize_sample(loader, num_sample):
    images, labels= next(iter(loader))
    fig, axes= plt.subplots(1,num_sample, figsize=(6,6))
    for i in range(num_sample):
        img= images[i].permute(2,1,0) # matplot -> h,w,c  normal -> c,h,w
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = (img * std) + mean
        img = torch.clamp(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"Labels:{labels[i].item()}")
        axes[i].axis("off")
        print(images[i].shape)
    plt.tight_layout()
    plt.show(block=True)