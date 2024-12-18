import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt


##########
def load_random_images_and_predictions(test_images_dir, predictions_dir):
    # List all subdirectories (folders) in the test images directory
    subfolders = [f.path for f in os.scandir(test_images_dir) if f.is_dir()]
    selected_folders = random.sample(subfolders, 3)  # Randomly select 3 folders
    
    images_and_predictions = []  # To store tuples of (image, prediction)

    for folder in selected_folders:
        # Get the image file (assuming only one image per folder)
        image_name = os.listdir(folder)[0]  # Get the image filename
        image_path = os.path.join(folder, image_name)

        # Construct the corresponding prediction path
        prediction_path = os.path.join(predictions_dir, image_name)

        # Load the image and prediction
        image      = Image.open(image_path).convert("RGB")
        prediction = Image.open(prediction_path) if os.path.exists(prediction_path) else None

        if image is not None and prediction is not None:
            images_and_predictions.append((image, prediction))
        else:
            print(f"Failed to load image or prediction for {image_name}")

    images_and_predictions_transposed = list(zip(*images_and_predictions)) # transpose images_and_pedictions s.t. images_and_prediction[0] are images, [1] are masks. Cannot use np.transpose since images_and_predictions is a list of tuples, images and masks have different shapes. 
    return images_and_predictions_transposed


def plot_ramdom_test_images(test_images_dir, predictions_dir):

    images_and_predictions = load_random_images_and_predictions(test_images_dir, predictions_dir)
    images      = images_and_predictions[0]
    predictions = images_and_predictions[1]

    plt.figure(figsize=(9, 9))
    idx = 0

    for image, prediction in zip(images, predictions) :

        plt.subplot(3, 3, 1 +3*idx)
        plt.imshow(image)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])  

        plt.subplot(3, 3, 3 +3*idx)
        plt.imshow(prediction, cmap="gray")
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])

        prediction = np.array(prediction)/255
        cmap = plt.cm.gray
        alpha = 0.5*prediction  

        plt.subplot(3, 3, 2 +3*idx)
        plt.imshow(image.resize((256, 256))) 
        plt.imshow(prediction, cmap=cmap, alpha=alpha) 
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])

        idx +=1

    plt.subplots_adjust(wspace=-0.2, hspace=0.2)  # Decrease the spacing
    plt.tight_layout()

##########

def main(test_images_dir, predictions_dir):

    plot_ramdom_test_images(test_images_dir, predictions_dir)
    plt.show()

if __name__ == "__main__" :

    test_images_dir = r'test_set_images'
    predictions_dir = r'predictions'

    main(test_images_dir=test_images_dir, predictions_dir=predictions_dir)