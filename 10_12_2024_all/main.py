import augment_data
import train 
import show_results

def main(original_images_dir,
         original_masks_dir,
         augmented_images_dir,
         augmented_masks_dir,
         test_images_dir,
         num_augment=30,            # number of augmented images per original image
         scale_limit=(-0.3, 1.5),    # scale of the zoom in the augmentation
         num_epochs=80,             # number of epoch to train 
         learning_rate=0.01,        # learning rate in the train
         batch_size=8,              # batch size in the train 
         threshold=0.5,             # threshold for the predictions
         ):
    
    model_name = f"UNet_4lev_Dice_norm_augmV3_{num_epochs}epochs_lr{learning_rate}_bs{batch_size}.pth"

    augment_data.main(input_images_dir  = original_images_dir,
                      input_masks_dir   = original_masks_dir,
                      output_images_dir = augmented_images_dir,
                      output_masks_dir  = augmented_masks_dir,
                      num_augment       = num_augment, 
                      scale_limit       = scale_limit)
    
    train.main(model_name       = model_name,
               train_image_dir  = augmented_images_dir,
               train_mask_dir   = augmented_masks_dir,
               num_epochs       = num_epochs,
               learning_rate    = learning_rate, 
               batch_size       = batch_size)
    
    show_results.main(test_images_dir   = test_images_dir,
                      threshold         = threshold)

if __name__ == '__main__':

    # The path of the directories where all the input images and masks are 
    original_images_dir = 'C:\\Users\\Gauthier\\Desktop\\EPFL\\Master\\Machine Learning\\projet\\project_2\\test\\original_data\\images\\'
    original_masks_dir = 'C:\\Users\\Gauthier\\Desktop\\EPFL\\Master\\Machine Learning\\projet\\project_2\\test\\original_data\\groundtruth\\'

    # the path of the directories of the augmented images / masks
    augmented_images_dir = 'C:\\Users\\Gauthier\\Desktop\\EPFL\\Master\\Machine Learning\\projet\\project_2\\test\\training\\all_images\\'
    augmented_masks_dir  = 'C:\\Users\\Gauthier\\Desktop\\EPFL\\Master\\Machine Learning\\projet\\project_2\\test\\training\\all_groundtruth\\'

    # the path to the test images
    test_images_dir = r'C:\Users\Gauthier\Desktop\EPFL\Master\Machine Learning\projet\project_2\test\test_set_images'

    main(original_images_dir,
         original_masks_dir,
         augmented_images_dir,
         augmented_masks_dir,
         test_images_dir,)