
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

""" 
WE MUST SEPERATE TRAIN SET INTO TRAIN AND VALIDATION ( HAVE A TRAIN AND A VALIDATION LOADER )
EACH EPOCH TEST THE VALIDATION TEST
MODEL IS SAVED FOR THE BEST VALIDATION LOSS
DON'T FORGET TO WRITE WITH TORCH NO GRAD FOR THE VALIDATION 
NO NEED TO DO BACKWARD PASS ON THE VALIDATION TEST 
THE IDEA IS THAT THE MODEL SHOULD NOT SEE THE VALIDATION MODEL AT ALL TO TRAIN
USUALL PROPORTIONS ARE 80% TRAIN / 20% VALIDATION
"""


def split_into_train_and_validation():
     return NotImplementedError




def train_epoch(model, dataloader, criterion, optimizer, epoch, num_epochs, device):

    epoch_start_time = time.time()

    model.train()
    epoch_loss = []

    model = model.to(device)

    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:

        for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images).squeeze(1)
                loss = criterion(outputs, masks.squeeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
 
    # Calculate epoch statistics
    mean_epoch_loss = np.mean(epoch_loss)
    epoch_duration = time.time() - epoch_start_time
    estimated_remaining_time = (num_epochs - epoch - 1) * epoch_duration

    # Print epoch summary
    print(f"Epoch [{epoch+1}/{num_epochs}] completed in {datetime.timedelta(seconds=int(epoch_duration))}.")
    print(f"Estimated time remaining: {datetime.timedelta(seconds=int(estimated_remaining_time))}")
    print(f"Mean Loss: {mean_epoch_loss:.4f}\n")

    return model, mean_epoch_loss



def train(model, dataloader, criterion, optimizer, num_epochs=40, device="cuda"):

    print(f"using {device} with {num_epochs} epochs.\n")

    best_loss = np.inf
    best_model_state = None
    best_epch = 0
    epoch_losses = []

    for epoch in range(num_epochs):
        model, mean_epoch_loss = train_epoch(model, dataloader, criterion, optimizer, epoch, num_epochs, device)

        epoch_losses.append(mean_epoch_loss)

        if mean_epoch_loss < best_loss:
            # updates best model info to later save the model with the best results
            best_epch = epoch+1
            best_loss = mean_epoch_loss
            best_model_state = model.state_dict()

    return model, best_model_state, best_epch, best_loss, epoch_losses
        

def main(model_name, train_image_dir, train_mask_dir, num_epochs, learning_rate, batch_size):

    print(f"model_name       : {model_name}")
    print(f"learning_rate    : {model_name}")
    print(f"batch_size       : {batch_size}")
    print(f"train images dir : {train_image_dir}")
    print(f"train masks dir  : {train_mask_dir}\n")

    # creating dataset as ImagesDataset object
    image_transform = transforms.Compose([  # transform for images
            transforms.Resize((256, 256)),  # redimension images to 256x256
            transforms.ToTensor()           # Convert to pytorch tensors
            ])
    mask_transform = transforms.Compose([   
            transforms.Resize((256, 256)),
            transforms.ToTensor()
            ])
    train_dataset = ImagesDataset(train_image_dir, train_mask_dir, image_transform=image_transform, mask_transform=mask_transform)

    # computing mean and std to normalize images only
    mean, std = train_dataset.get_mean_std()
    
    # normalization is added to the pre-existing list of image transforms
    train_dataset.image_transform =  transforms.Compose([
                                        train_dataset.image_transform, # already existing transforms 
                                        transforms.Normalize(mean=mean, std=std) # normailzation added
                                        ])
    
    # the loader used in the training
    pin_memory = torch.cuda.is_available()  # pin_memory True only if using cuda 
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

    # set model architecture, loss and optimizer
    model = UNet()
    criterion = DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # the training
    train_start_time = datetime.datetime.now()
    print(f"Training started at: {train_start_time}")

    
    _, best_model_state, best_epch, loss, epoch_losses = train(model, train_loader, criterion, optimizer, num_epochs)

    train_end_time = datetime.datetime.now()
    print(f"\nTraining ended at: {train_end_time}")

    duration = train_end_time - train_start_time
    print(f"Training duration: {duration}")

    # saving the model
    torch.save({
        'model_state_dict': best_model_state,
        'num_epochs' : num_epochs,
        'best_epch' : best_epch,
        'mean': mean,
        'std' : std,
        'loss': loss,
        'epoch_losses' : epoch_losses,
        'batch_size' : batch_size,
        'learning_rate' : learning_rate,
        'training_duration' : duration,
        'image_transform' : train_dataset.image_transform,
        }, model_name)
    
    print("\nModel saved as : ",model_name)


if __name__ == "__main__":
     
    train_image_dir = 'C:\\Users\\Gauthier\\Desktop\\EPFL\\Master\\Machine Learning\\projet\\project_2\\test\\training\\all_images'
    train_mask_dir = 'C:\\Users\\Gauthier\\Desktop\\EPFL\\Master\\Machine Learning\\projet\\project_2\\test\\training\\all_groundtruth'
     
    num_epochs = 80
    learning_rate = 0.01
    batch_size = 8
    model_name = f"UNet_4lev_Dice_norm_augmV3_{num_epochs}epochs_lr{learning_rate}_bs{batch_size}.pth"

    main(model_name=model_name,
         train_image_dir=train_mask_dir,
         train_mask_dir=train_mask_dir,
         num_epochs=num_epochs,
         learning_rate=learning_rate,
         batch_size=batch_size)
    

