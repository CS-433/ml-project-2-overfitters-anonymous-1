import numpy as np
import re
from PIL import Image

# assign a label to a patch
def patch_to_label(patch, foreground_threshold):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    

def mask_to_submission_strings(image_filename, foreground_threshold, desired_shape=(608, 608)):
    """Reads a single image, resizes it, and outputs the strings for the submission file."""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = Image.open(image_filename).convert("L")  # Convert to grayscale if needed
    im_resized = im.resize(desired_shape, Image.NEAREST)  # Resize using nearest neighbor interpolation
    im_resized = np.array(im_resized)  # Convert to NumPy array for processing

    patch_size = 16
    for j in range(0, im_resized.shape[1], patch_size):
        for i in range(0, im_resized.shape[0], patch_size):
            patch = im_resized[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, foreground_threshold)
            yield("{:03d}_{}_{},{}".format(img_number,j,i,label))


def masks_to_submission(submission_filename, foreground_threshold, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, foreground_threshold))


def main(submission_filename, foreground_threshold):
    image_filenames = []
    for i in range(1, 51):
        image_filename = 'predictions/' + 'test_%.1d' % i + '.png' 
        print(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, foreground_threshold, *image_filenames)


if __name__ == '__main__':

    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    output_name = 'Predictions_submission.csv' # Folder / name of the output file

    main(submission_filename=output_name, foreground_threshold=foreground_threshold)
