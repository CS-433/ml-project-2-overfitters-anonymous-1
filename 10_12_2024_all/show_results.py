from UNet import UNet
import torch
from torchvision import transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt


##########

def load_random_images(test_images_dir, num_images=3):
    # List all subdirectories (folders) in the test images directory
    subfolders = [f.path for f in os.scandir(test_images_dir) if f.is_dir()]
    selected_folders = random.sample(subfolders, num_images) # Randomly select 'num_images' folders
    
    images = []
    for folder in selected_folders:
        # Get the image file (assuming only one image per folder)
        image_path = os.path.join(folder, os.listdir(folder)[0])
        image = Image.open(image_path).convert("RGB")
        
        if image is not None:
            images.append(image)
        else:
            print(f"Failed to load image from {image_path}")
    
    return images


def plot_ramdom_test_images(test_images_dir, model, transform, device, threshold):

    random_images = random_images = load_random_images(test_images_dir, num_images=3)
    random_cuda_images  = random_images
    all_predicted_masks = []

    plt.figure(figsize=(9, 9))
    idx = 0

    for image, cuda_image in zip(random_images, random_cuda_images) :
        cuda_image = transform(image).unsqueeze(0)
        cuda_image = cuda_image.to(device)

        # make a prediction
        with torch.no_grad():
            output = model(cuda_image) 
            prediction = torch.sigmoid(output).squeeze(0)
            mask = prediction > threshold

        # Convertmask to numpy to be displayed
        mask = mask.cpu().numpy()
        mask = mask.squeeze(0)
        all_predicted_masks.append(mask)

        plt.subplot(3, 3, 1 +3*idx)
        plt.imshow(image)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])  

        plt.subplot(3, 3, 3 +3*idx)
        plt.imshow(mask, cmap="gray")
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])

        mask = mask.astype(float)
        cmap = plt.cm.gray
        alpha = 0.5*mask  

        plt.subplot(3, 3, 2 +3*idx)
        plt.imshow(image.resize((256, 256))) 
        plt.imshow(mask, cmap=cmap, alpha=alpha) 
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])

        idx +=1

    plt.subplots_adjust(wspace=-0.2, hspace=0.2)  # Decrease the spacing
    plt.tight_layout()

##########

def main(test_images_dir, threshold):
    # Load the saved model checkpoint
    model_path = r"C:\Users\Gauthier\Desktop\EPFL\Master\Machine Learning\projet\project_2\test\UNet_4lev_Dice_norm_augmV2_79epochs.pth"
    checkpoint = torch.load(model_path)

    # Extract the model state, normalization parameters, and best loss
    model_state_dict  = checkpoint['model_state_dict']
    # num_epcohs        = checkpoint['num_epochs']
    # best_epoch        = checkpoint['best_epoch']
    mean              = checkpoint['mean']
    std               = checkpoint['std']
    # loss              = checkpoint['loss']
    # epoch_losses      = checkpoint['epoch_losses']
    # batch_size        = checkpoint['batch_size']
    # learning_rate     = checkpoint['learning_rate']
    training_duration = checkpoint['training_duration']
    # image_transform   = checkpoint['image_transform']

    print(f'training_duration : {training_duration}')

    # Load the model state
    model = UNet()
    model.load_state_dict(model_state_dict)
    model.eval()

    # test_images_dir  = r"C:\Users\Gauthier\Desktop\EPFL\Master\Machine Learning\projet\project_2\test\test_set_images"
    # train_images_dir = r"C:\Users\Gauthier\Desktop\EPFL\Master\Machine Learning\projet\project_2\test\training\all_images"
    # train_masks_dir  = r"C:\Users\Gauthier\Desktop\EPFL\Master\Machine Learning\projet\project_2\test\training\all_groundtruth"


    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),   
        transforms.Normalize(mean=mean, std=std) 
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = model.to(device)

    plot_ramdom_test_images(test_images_dir,trained_model,transform,device,threshold)
    plt.show()

if __name__ == "__main__" :
    test_images_dir = r'C:\Users\Gauthier\Desktop\EPFL\Master\Machine Learning\projet\project_2\test\test_set_images'
    main(test_images_dir, threshold=0.5)

    


    


