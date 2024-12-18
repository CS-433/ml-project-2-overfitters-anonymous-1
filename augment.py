import os
import cv2
from tqdm import tqdm
import albumentations as A
import files_helpers as fh

def augment_image_and_mask(image, mask, scale_limits):
    """
    Applies basic augmentations to an image and mask for aerial segmentation.

    Args:
        image (np.array): Input RGB image.
        mask (np.array): Corresponding grayscale mask.
        scale_limits (tuple): Scaling factor range. If scale_limits is a single float value, 
                              the range will be (-scale_limits, scale_limits).

    Returns:
        tuple: Augmented image and mask.
    """
    # Define augmentation pipeline
    aug_pipeline = A.Compose([
        A.RandomBrightnessContrast(p=0.2),                 # Apply brightness/contrast only to the image
        A.HorizontalFlip(p=0.5),                           # Flip both image and mask
        A.ShiftScaleRotate(scale_limit=scale_limits, rotate_limit=180, p=1)  # Scale-Rotate
    ], additional_targets={'mask': 'mask'})                # Ensure the mask is properly augmented

    # Apply augmentations
    augmented = aug_pipeline(image=image, mask=mask)
    return augmented['image'], augmented['mask']



def main(input_images_dir, input_masks_dir, output_images_dir, output_masks_dir, num_augment, scale_limit):

    print("\nAugmenting images sequence started.\n")
    
    all_input_images_paths = os.listdir(input_images_dir)
    all_input_masks_paths = os.listdir(input_masks_dir)

    # remove then create needed folders to ensure augmented images are new and in right number
    fh.remove_folder(output_images_dir)
    fh.remove_folder(output_masks_dir)
    fh.create_folder(output_images_dir)
    fh.create_folder(output_masks_dir)

    for image_name, mask_name in tqdm(zip(all_input_images_paths, all_input_masks_paths), # tqdm is just to have a progression bar
                                    total=len(all_input_images_paths), 
                                    desc="Augmenting images"):

        # the full path to the image / mask
        image_path = os.path.join(input_images_dir, image_name)
        mask_path  = os.path.join(input_masks_dir,   mask_name)

        # gives an image ( numpy array ) from the image path
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None: # sanity chek, raise an error if the image is not found. 
            raise FileNotFoundError(f"These files cannot be found :\nImage: {image_path} \nMask: {mask_path} \nMay have a wrong path.")

        for n in range(num_augment):

            augmented_image, augmented_mask = augment_image_and_mask(image, mask,scale_limit)

            output_image_path = os.path.join(output_images_dir, image_name[:-4] + '_augmentation_' + str(n+1) + '.png')
            output_mask_path  = os.path.join(output_masks_dir,  mask_name[:-4]  + '_augmentation_' + str(n+1) + '.png')
            
            cv2.imwrite(output_image_path, augmented_image)
            cv2.imwrite(output_mask_path, augmented_mask)

    # copies the 100 original images and masks to the training folders
    fh.copy_images(input_images_dir, output_images_dir)
    fh.copy_images(input_masks_dir,  output_masks_dir )


if __name__ == "__main__":
    # The path of the directories where all the input images and masks are 
    input_images_dir = r'original_data\images'
    input_masks_dir = r'original_data\groundtruth'

    # the path of the directories of the augmented images / masks
    output_images_dir = r'training\all_images'
    output_masks_dir  = r'training\all_groundtruth'

    main(input_images_dir=input_images_dir,
         input_masks_dir=input_masks_dir,
         output_images_dir=output_images_dir,
         output_masks_dir=output_masks_dir,
         num_augment=30,
         scale_limit=(-0.3, 1.5), )