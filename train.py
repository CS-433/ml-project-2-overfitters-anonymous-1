
from UNet import UNet
from ImagesDataset import ImagesDataset

from tqdm import tqdm 

import numpy as np
import torch
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader 
from torchvision import transforms 


def train_epoch(model, dataloader, criterion, optimizer, epoch, num_epochs, device):

    model.train()
    epoch_loss = []

    model = model.to(device)

    for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, masks.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

    mean_epoch_loss = np.mean(epoch_loss)  
    print(f"Epoch [{epoch+1}/{num_epochs}], Mean Loss: {mean_epoch_loss:.4f}") 
    return model



def train(model, dataloader, criterion, optimizer, num_epochs=40, device="cpu"):

    for epoch in range(num_epochs):
        model = train_epoch(model, dataloader, criterion, optimizer, epoch, num_epochs, device)

    return model
        

if __name__ == "__main__":
    
    train_image_dir = "./training/images/"  # Images d'entraînement
    train_mask_dir = "./training/groundtruth/"  # Masques d'entraînement

    # Transformations des données (mise à l'échelle et conversion en tenseurs)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Redimensionner les images à 256x256
        transforms.ToTensor()  # Convertir en tenseur PyTorch
    ])


    train_dataset = ImagesDataset(train_image_dir, train_mask_dir, transform)
    train_loader  = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = UNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # training
    num_epochs = 30
    trained_model = train(model, train_loader, criterion, optimizer, num_epochs)

    model_name = "UNet_4levels_" + str(num_epochs) + "epochs.pth"
    torch.save(trained_model.state_dict(), model_name)
    print("Modèle sauvegardé sous ",model_name)
