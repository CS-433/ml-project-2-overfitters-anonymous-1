from UNet4 import UNet4
from UNet5 import UNet5
import torch
from torchvision import transforms
from PIL import Image
import os
import files_helpers as fh

def prediction_from_image(trained_model, threshold, transform, image_path, predictions_dir, device):

    img = Image.open(image_path).convert("RGB")

    cuda_image = transform(img).unsqueeze(0)
    cuda_image = cuda_image.to(device)

    # make a prediction
    with torch.no_grad():
        output = trained_model(cuda_image) 
        prediction = torch.sigmoid(output).squeeze(0)
        prediction_mask = prediction > threshold

    # Convert mask to numpy to be displayed
    prediction_mask = prediction_mask.cpu().numpy()
    prediction_mask = prediction_mask.squeeze(0)

    # Save prediction with the same name as the input image
    prediction_name = os.path.basename(image_path)  # Extract image name
    prediction_path = os.path.join(predictions_dir, prediction_name)


    # save also  in png file, for visualization
    prediction_mask_image = Image.fromarray((prediction_mask * 255).astype('uint8'))
    prediction_mask_image.save(prediction_path)
    
    # csv_helpers.create_csv_submission()
    return prediction_mask


def main(model_name, test_images_dir, predictions_dir, threshold):
    # Load the saved model checkpoint
    checkpoint = torch.load(model_name)
    print(f"\nMaking predictions using the model {model_name}\n")

    # Extract the model state, etc
    model_state_dict  = checkpoint['model_state_dict']
    mean              = checkpoint['mean']
    std               = checkpoint['std']
    n_levels          = checkpoint.get("n_levels", 4)

    transform =  transforms.Compose([transforms.Resize((256, 256)),  
                                     transforms.ToTensor(),   
                                     transforms.Normalize(mean=mean, std=std) 
                                     ])
    
    # Load the model state
    if n_levels == 5:
        model = UNet5()
    else :
        model = UNet4()

    model.load_state_dict(model_state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = model.to(device)

    # creates the folder for the predicted masks
    fh.remove_folder(predictions_dir)
    fh.create_folder(predictions_dir)

    subfolders = [f.path for f in os.scandir(test_images_dir) if f.is_dir()]

    for subfolder in subfolders:
        all_images_paths = [os.path.join(subfolder, file) for file in os.listdir(subfolder) if file.endswith('png')]
        
        for image_path in all_images_paths:
            prediction_from_image(trained_model, threshold, transform, image_path, predictions_dir, device)
            
            # print(f"Prediction for {prediction_name} saved to {prediction_path}")
    print(f"All predictions saved to {predictions_dir}\n")

if __name__ == "__main__":
    model_name = r'UNet4_augm60_epochs50_lr0.01_bs8_wd0.pth'
    test_images_dir = r'test_set_images'
    predictions_dir = r'predictions'
    threshold = 0.5
    main(model_name, test_images_dir, predictions_dir, threshold)