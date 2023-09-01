import torch
import torch.nn as nn
import torchvision.models as models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from torchvision.models import resnet18
from torchsummary import summary

# Define the Generator model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet18 = resnet18(progress=True)
        self.resnet18.fc = nn.Sequential(nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.resnet18(x)
        return y

class UltrasoundDataset(Dataset):
    def __init__(self, low_folder, high_folder, transform=None):
        self.low_folder = low_folder
        self.high_folder = high_folder
        self.low_images = os.listdir(low_folder)
        self.high_images = os.listdir(high_folder)
        self.transform = transform
        self.imgs = self.low_images + self.high_images
        for i in range(len(self.imgs)):
            if i < len(self.low_images):
                self.imgs[i] = os.path.join(self.low_folder, self.imgs[i])
                
            else:
                self.imgs[i] = os.path.join(self.high_folder, self.imgs[i])
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        
        img = Image.open(img_name).convert('RGB')
        
        if img_name.split('/')[-2] == 'A':
            label = torch.as_tensor(0.)
        else:
            label = torch.as_tensor(1.)
        
        if self.transform:
            img = self.transform(img)
        
        sample = {'image': img.to('cuda'), 'label': label.to('cuda')}
        return sample

if __name__ == '__main__':
    
    # Paths to your low and high ultrasound image folders
    low_folder = r'/home/engssrd/Desktop/Challenge_MICCAI/uvcgan2-main/uvcgan2/data/datasets/Challenge_MICCAI/All_folds/train/A'
    high_folder = r'/home/engssrd/Desktop/Challenge_MICCAI/uvcgan2-main/uvcgan2/data/datasets/Challenge_MICCAI/All_folds/train/B'
    save_path = r'/home/engssrd/Desktop/Challenge_MICCAI/Ultrasound_perceptual_network/Model'
    num_epochs = 1000
    
    # Define data transformations if needed (e.g., resizing, normalization)
    data_transform = transforms.Compose([
        transforms.Resize((286, 286)), transforms.RandomCrop(256),
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create the  dataset
    dataset = UltrasoundDataset(low_folder, high_folder, transform=data_transform)
    
    # Create a DataLoader for batch processing
    batch_size = 8
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    generator = Classifier().to('cuda')
    loss_MSE = nn.MSELoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    
    summary(generator, torch.zeros(1,3,256,256))
    # Training loop
    for epoch in range(num_epochs):
        step = 0
        for sample in data_loader:
            step += 1
            optimizer.zero_grad()
            
            # Generate high ultrasound images from low ultrasound images
            y_pred = generator(sample['image'])
            
            # Compute perceptual loss
            loss = loss_MSE(y_pred.squeeze(), sample['label'])
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            # Print progress
            if (step + 1) % 50 == 0:  # Print every 10 iterations
                print(f"Epoch {epoch+1} - Iteration [{step+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
    
    
    # Save the trained generator
    torch.save(generator.state_dict(), os.path.join(save_path,'generator.pth'))

