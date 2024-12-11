import os
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class ImagesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        """
        Initialise le dataset.
        - image_dir : répertoire contenant les images satellites
        - mask_dir : répertoire contenant les masques binaires (ground truth)
        - transform : transformations à appliquer sur les images et masques
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform  = mask_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        """
        Retourne le nombre d'images dans le dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Charge une image et son masque correspondant à l'index donné.
        """
        # Obtenir le chemin de l'image et du masque
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Charger l'image (RGB) et le masque (grayscale)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Appliquer les transformations, si spécifiées
        if self.image_transform :
            image = self.image_transform(image)
        if self.mask_transform :
            mask = self.mask_transform(mask)
            mask = (mask > 0).float()  # Convertir le masque en binaire (1 pour route, 0 pour arrière-plan)

        return image, mask  # Retourne l'image et son masque
    

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
            images = images.view(batch_samples, 3, -1)  # Flatten HxW
            mean += images.mean(2).sum(0)  # Sum across batches
            std += images.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std  /= nb_samples

        print(f"Mean : {mean} \nStd  : {std} \n")
        return mean, std
    
if __name__ == "__main__":
    images_dir = "C:\\Users\\Gauthier\\Desktop\\EPFL\\Master\\Machine Learning\\projet\\project_2\\test\\training\\augmented_images"
    masks_dir  = "C:\\Users\\Gauthier\\Desktop\\EPFL\\Master\\Machine Learning\\projet\\project_2\\test\\training\\augmented_groundtruth"

    test_dataset = ImagesDataset(image_dir=images_dir, mask_dir=masks_dir, image_transform=transforms.ToTensor())
    test_dataset.get_mean_std()