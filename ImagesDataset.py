import os
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class ImagesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        """
        Initialize dataset.
        - image_dir : directory containing the images
        - mask_dir : directory containing the ground truths
        - transform : transforms to be applied on images and masks
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform  = mask_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        """
        returns number of images in the dataset
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        loads an image and the corresponding masks at a given index
        """

        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Applying transforms if any
        if self.image_transform :
            image = self.image_transform(image)
        if self.mask_transform :
            mask = self.mask_transform(mask)
            mask = (mask > 0).float()  

        return image, mask  
    

    def get_mean_std(self, device="cuda"): 
        if device == "cuda": # uses GPU if possible
            temp_loader = DataLoader(self, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        else :
            temp_loader = DataLoader(self, batch_size=64, shuffle=False, num_workers=0)

        # Compute mean and standard deviation
        mean = 0.0
        std = 0.0
        nb_samples = 0

        for images, _ in tqdm(temp_loader, total=len(temp_loader), desc="Computing Mean and Std"):  # Only process images
            images = images.to(device)
            batch_samples = images.size(0)  # Batch size
            images = images.view(batch_samples, 3, -1)  
            mean += images.mean(2).sum(0)  # Sum across batches
            std += images.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std  /= nb_samples

        print(f"Mean : {mean} \nStd  : {std} \n")
        return mean, std
