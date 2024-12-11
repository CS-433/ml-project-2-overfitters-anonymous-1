from torch.utils.data import random_split, DataLoader
from UNet import UNet
from ImagesDataset import ImagesDataset

from tqdm import tqdm 
import time
import datetime

import numpy as np
import torch
import torch.optim as optim  
from torch.utils.data import DataLoader 
from torchvision import transforms 
from segmentation_models_pytorch.losses import DiceLoss


def train_epoch(model, train_loader, val_loader, criterion, optimizer, epoch, num_epochs, device):
    epoch_start_time = time.time()

    # training phase ###
    model.train()
    train_loss = []
    model = model.to(device)

    with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - train      ", unit="batch") as pbar:
        for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images).squeeze(1)
                loss = criterion(outputs, masks.squeeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

    mean_train_loss = np.mean(train_loss)

    # validation phase ###
    model.eval()
    val_loss = []
    with torch.no_grad():                                                                  
        for images, masks in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - validation " ):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, masks.squeeze(1))
            val_loss.append(loss.item())

    mean_val_loss = np.mean(val_loss)
 
    # Calculate time estimations
    epoch_duration = time.time() - epoch_start_time
    estimated_remaining_time = (num_epochs - epoch - 1) * epoch_duration

    # Print epoch summary
    print(f"Epoch [{epoch+1}/{num_epochs}] completed in {datetime.timedelta(seconds=int(epoch_duration))}.")
    print(f"Estimated time remaining : {datetime.timedelta(seconds=int(estimated_remaining_time))}")
    print(f"Mean train Loss : {mean_train_loss:.4f}, Mean validation loss : {mean_val_loss:.4f}\n")

    return model, mean_train_loss, mean_val_loss



def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=40, device="cuda"):

    print(f"using {device} with {num_epochs} epochs.\n")

    best_mean_val_loss = np.inf
    best_model_state = None
    best_epoch = 0
    all_train_losses = []
    all_val_losses   = []

    for epoch in range(num_epochs):
        model, mean_train_loss, mean_val_loss = train_epoch(model, train_loader, val_loader, criterion, optimizer, epoch, num_epochs, device)

        all_val_losses.append(mean_val_loss)
        all_train_losses.append(mean_train_loss)

        if mean_val_loss < best_mean_val_loss:
            # updates best model info to later save the model with the best results
            best_epoch = epoch+1
            best_mean_val_loss = mean_val_loss
            best_model_state = model.state_dict()

    return model, best_model_state, best_epoch, best_mean_val_loss, all_val_losses, all_train_losses
        

def main(model_name, train_image_dir, train_mask_dir, num_epochs, learning_rate, weight_decay, batch_size):

    print(f"model_name       : {model_name}")
    print(f"learning_rate    : {learning_rate}")
    print(f"batch_size       : {batch_size}")
    print(f"weight decay     : {weight_decay}")
    print(f"train images dir : {train_image_dir}")
    print(f"train masks dir  : {train_mask_dir}\n")

    # Create dataset
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Full dataset
    full_dataset = ImagesDataset(train_image_dir, train_mask_dir, 
                                image_transform=image_transform, 
                                mask_transform=mask_transform)

    # Compute mean and std for normalization
    mean, std = full_dataset.get_mean_std()

    # Update image transforms to include normalization ( only images are normalized, not masks )
    image_transform_with_norm = transforms.Compose([
        image_transform,                          # Existing image transforms
        transforms.Normalize(mean=mean, std=std)  # Add normalization
    ])

    # Update the dataset's transform with normalization
    full_dataset.image_transform = image_transform_with_norm

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))  # 80% for training
    val_size = len(full_dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # set model architecture, loss and optimizer
    model = UNet()
    criterion = DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # do we want weight decay ? 

    # the training
    train_start_time = datetime.datetime.now()
    print(f"Training started at: {train_start_time}")
    
    _, best_model_state, best_epoch, best_mean_val_loss, all_val_losses, all_train_losses = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    train_end_time = datetime.datetime.now()
    print(f"\nTraining ended at: {train_end_time}")

    duration = train_end_time - train_start_time
    print(f"Training duration: {duration}")

    # saving the model
    torch.save({
        'model_state_dict'  : best_model_state,
        'num_epochs'        : num_epochs,
        'best_epoch'        : best_epoch,
        'mean'              : mean,
        'std'               : std,
        'val_loss'          : best_mean_val_loss,
        'all_val_losses'    : all_val_losses,
        'all_train_losses'  : all_train_losses,
        'batch_size'        : batch_size,
        'learning_rate'     : learning_rate,
        'weight_decay'      : weight_decay,
        'training_duration' : duration,
        'image_transform'   : full_dataset.image_transform,
        }, model_name)
    
    print("\nModel saved as : ",model_name,"\n")


if __name__ == "__main__":
     
    train_image_dir = 'C:\\Users\\Gauthier\\Desktop\\EPFL\\Master\\Machine Learning\\projet\\project_2\\test\\training\\all_images'
    train_mask_dir = 'C:\\Users\\Gauthier\\Desktop\\EPFL\\Master\\Machine Learning\\projet\\project_2\\test\\training\\all_groundtruth'
     
    num_epochs = 1
    learning_rate = 0.01
    batch_size = 8
    weight_decay = 0
    model_name = f"test_UNet_4lev_Dice_norm_augmV3_split_{num_epochs}epochs.pth"

    main(model_name      = model_name,
         train_image_dir = train_image_dir,
         train_mask_dir  = train_mask_dir,
         num_epochs      = num_epochs,
         learning_rate   = learning_rate,
         weight_decay    = weight_decay,
         batch_size      = batch_size)