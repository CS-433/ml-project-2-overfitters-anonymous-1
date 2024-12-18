import augment
import train 
import predict
import submit
import show

def main(original_images_dir,
         original_masks_dir,
         augmented_images_dir,
         augmented_masks_dir,
         test_images_dir,
         predictions_dir,
         n_levels=4,                # levels in the UNet, 4 or 5, but 5 is extremely slow 
         num_augment=60,            # number of augmented images per original image
         scale_limit=(-0.4, 1.6),   # scale of the zoom in the augmentation
         num_epochs=50,             # number of epoch to train 
         learning_rate=0.01,        # learning rate in the train
         weight_decay=0,            # weight decay for the adam optimizer  in the train
         batch_size=8,              # batch size in the train
         threshold=0.5,             # threshold for the predictions
         ):
     
     model_name = f"UNet{n_levels}_augm{num_augment}_epochs{num_epochs}_lr{learning_rate}_bs{batch_size}_wd{weight_decay}.pth"
     submission_filename = model_name[:-4] + "_submission.csv"

     # augment images
     augment.main(input_images_dir      = original_images_dir,
                      input_masks_dir   = original_masks_dir,
                      output_images_dir = augmented_images_dir,
                      output_masks_dir  = augmented_masks_dir,
                      num_augment       = num_augment, 
                      scale_limit       = scale_limit)
    
    # create train and save model
     train.main(model_name       = model_name,
                train_image_dir  = augmented_images_dir,
                train_mask_dir   = augmented_masks_dir,
                n_levels         = n_levels,
                num_epochs       = num_epochs,
                learning_rate    = learning_rate, 
                weight_decay     = weight_decay,
                batch_size       = batch_size,
                n_augment        = num_augment)
     
     # create png images of the masks predicted by the model
     predict.main(model_name      = model_name,
                  test_images_dir = test_images_dir,
                  predictions_dir = predictions_dir,
                  threshold       = threshold)
     
     # create csv file from the png masks in the right format for submission
     submit.main(submission_filename  = submission_filename,
                 foreground_threshold = 0.25)

     # show randomly three test images and their predicted masks
     show.main(test_images_dir = test_images_dir,
               predictions_dir = predictions_dir)

if __name__ == '__main__':

     # The path of the directories where all the input images and masks are 
     original_images_dir = r'original_data\images'
     original_masks_dir  = r'original_data\groundtruth'

     # the path of the directories of the augmented images / masks
     augmented_images_dir = r'training\all_images'
     augmented_masks_dir  = r'training\all_groundtruth'

     # the path to the test images
     test_images_dir = r'test_set_images'

     # the path to the directory where predictions will be saved
     predictions_dir = r'predictions'

     main(original_images_dir  = original_images_dir,
          original_masks_dir   = original_masks_dir,
          augmented_images_dir = augmented_images_dir,
          augmented_masks_dir  = augmented_masks_dir,
          test_images_dir      = test_images_dir,
          predictions_dir      = predictions_dir)
