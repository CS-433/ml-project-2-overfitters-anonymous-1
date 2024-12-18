from UNet5 import UNet5
from UNet4 import UNet4
from ImagesDataset import ImagesDataset

from tqdm import tqdm 
import time
import datetime

import numpy as np

import torch
import torch.optim as optim  
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader
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



def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=40, device="cuda"):

    print(f"using {device} with {num_epochs} epochs.\n")

    best_mean_val_loss = np.inf
    best_model_state = None
    best_epoch = 0
    all_train_losses   = []
    all_val_losses     = []
    all_learning_rates = []

    for epoch in range(num_epochs):
        model, mean_train_loss, mean_val_loss = train_epoch(model, train_loader, val_loader, criterion, optimizer, epoch, num_epochs, device)

        all_val_losses.append(mean_val_loss)
        all_train_losses.append(mean_train_loss)

        if mean_val_loss < best_mean_val_loss:
            # updates best model info to later save the model with the best results
            best_epoch = epoch+1
            best_mean_val_loss = mean_val_loss
            best_model_state = model.state_dict()
        
        # update the learning rate
        scheduler.step(mean_val_loss)
        # Store the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        all_learning_rates.append(current_lr)

    return model, best_model_state, best_epoch, best_mean_val_loss, all_val_losses, all_train_losses, all_learning_rates
        

def main(model_name, train_image_dir, train_mask_dir, n_levels, num_epochs, learning_rate, weight_decay, batch_size, n_augment):

    # some prints to check if model has the desired parameters before letting it train for hours
    print(f"model_name       : {model_name}")
    print(f"n_levels         : {n_levels}")
    print(f"learning_rate    : {learning_rate}")
    print(f"batch_size       : {batch_size}")
    print(f"weight decay     : {weight_decay}")
    print(f"train images dir : {train_image_dir}")
    print(f"train masks dir  : {train_mask_dir}\n")

    # mak and images have different transforms : we don't want the maks to be normaized
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # the full dataset, will be split in train and validation
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
    pin_memory = torch.cuda.is_available()  # faster if using GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # set model architecture
    if n_levels == 5:
        model = UNet5() # turns out 5 levels is computationaly too expensive
    elif n_levels == 4:
        model = UNet4()
    else :
        model = UNet4()
        print(f"\nUnet with {n_levels} levels is not implemented - model is build with 4 levels by default.\n")

    criterion = DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-4)    # learning rates decades by *0.1 when loss hits a plateau on a given number of consecutive epochs.

    # the training ###
    train_start_time = datetime.datetime.now()
    print(f"Training started at: {train_start_time}")
    
    _, best_model_state, best_epoch, best_mean_val_loss, all_val_losses, all_train_losses, all_learning_rates = train(model,train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)

    train_end_time = datetime.datetime.now()
    print(f"\nTraining ended at: {train_end_time}")

    duration = train_end_time - train_start_time
    print(f"Training duration: {duration}")

    # saving the model ###
    # all variables are saved to use them later to make plots
    torch.save({
        'model_state_dict'  : best_model_state,
        'n_levels'          : n_levels,
        'num_epochs'        : num_epochs,
        'best_epoch'        : best_epoch,
        'mean'              : mean,
        'std'               : std,
        'val_loss'          : best_mean_val_loss,
        'all_val_losses'    : all_val_losses,
        'all_train_losses'  : all_train_losses,
        'all_learning_rates': all_learning_rates,
        'batch_size'        : batch_size,
        'learning_rate'     : learning_rate,
        'weight_decay'      : weight_decay,
        'training_duration' : duration,
        'image_transform'   : full_dataset.image_transform,
        'n_aumgent'         : n_augment,
        }, model_name)
    
    print("\nModel saved as : ",model_name,"\n")


if __name__ == "__main__":
     
    train_image_dir = r'training\all_images'
    train_mask_dir = r'training\all_groundtruth'
     
    num_epochs = 1
    learning_rate = 0.1
    batch_size = 8
    weight_decay = 0
    n_levels = 4
    n_augment = 30
    model_name = f"UNet{n_levels}_epochs{num_epochs}_lr{learning_rate}_bs{batch_size}_wd{weight_decay}.pth"

    main(model_name      = model_name,
         train_image_dir = train_image_dir,
         train_mask_dir  = train_mask_dir,
         n_levels        = n_levels,
         num_epochs      = num_epochs,
         learning_rate   = learning_rate,
         weight_decay    = weight_decay,
         batch_size      = batch_size,
         n_augment       = n_augment)