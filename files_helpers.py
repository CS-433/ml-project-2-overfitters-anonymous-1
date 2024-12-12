import os
import shutil

def create_folder(folder_path):
    # creates a folder, if it does not already exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"folder '{folder_path}' created.")
    else:
        print(f"folder '{folder_path}' already exists.")


def remove_folder(folder):
    if os.path.isdir(folder):
        try:
            shutil.rmtree(folder)  # Remove the folder and all its contents
            print(f"Folder '{folder}' removed.")
        except Exception as e:
            print(f"Error while removing folder '{folder}': {e}")
    else:
        print(f"'{folder}' is not a valid directory.")



def copy_images(input_images_dir, output_images_dir):
    # copy all images from input_images_dir to output_images_dir
    count = 0
    for file_name in os.listdir(input_images_dir):
        source_path      = os.path.join(input_images_dir, file_name)
        destination_path = os.path.join(output_images_dir, file_name)
        if os.path.isfile(source_path) and file_name.lower().endswith('.png'): # copy file only if it is a png image 
            destination_path = os.path.join(output_images_dir, file_name)
            shutil.copy2(source_path, destination_path)  # Copy the file, overwrite if file with same name 
            count += 1
    print(f'\n{count} images where successfully copied, \nfrom {input_images_dir} \nto {output_images_dir}.\n')