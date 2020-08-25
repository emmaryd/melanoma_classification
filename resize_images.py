import os
import sys
from PIL import Image
from tqdm import tqdm

def resize_from_dir(src, dest, shape=(256, 256)):
    """ Resize the images.

    Args:
        src: A string that is the path to the directory where the images are located.
        dest: A string that is the path to the directory where the resized images will be saved.
        shape: A tuple giving the size of the resized image.
    """
    os.makedirs(dest, exist_ok=True)
    for img_name in tqdm(os.listdir(src)):
        img_path = os.path.join(src, img_name)
        image = Image.open(img_path)
        image = image.resize(shape, Image.ANTIALIAS)
        
        dest_img_name = os.path.join(dest, img_name)
        image.save(dest_img_name)

if __name__ == "__main__":
    resize_from_dir(*sys.argv[1:])
