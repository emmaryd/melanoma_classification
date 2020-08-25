# melanoma_classification
Classification of malignt vs benignt melanoma using the ISIC 2020 Challenge Dataset, https://doi.org/10.34970/2020-ds01 (c) by ISDIS, 2020.
This Dataset can be found and downloaded at [https://www.kaggle.com/c/siim-isic-melanoma-classification/data].

## How to run:
The original images needs to be resized to reduce the computational time, this is done with `resize_images.py`, the images can be reduced to desired shape.

Before running the main file, the csv files needs to be modified. This is done using the `prepare_data.py` file. To run this, use the  `train.csv`, `test.csv` downloaded at the link above. And `duplicates_csv` is provided in the folder `data`. Running this script will create the processed train and test csv (`train_processed.csv` and `test_processed.csv`) that is needed to run `main.py`. 

To run the main function, `main.py`, the paths: `train_img_path` and `test_img_path` should be changed to the path to the directory that contains the resized train and test images.
Also, the data_train and test_train should be change to the path to the proessed train and test csv files are located.
