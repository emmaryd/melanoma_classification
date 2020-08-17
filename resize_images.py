import os
from PIL import Image
import sys
from tqdm import tqdm
def resize_from_dir(src, dest, shape=(256,256)):
    """
    Resizes the images in the source directory "scr" to size=(256,256).
    The resized images gets saved in "dest" folder. 
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
    #resize_from_dir(sys.argv[1], sys.argv[2])
