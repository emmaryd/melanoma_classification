# melanoma_classification
Classification of malignt vs benignt melanoma using the ISIC 2020 Challenge Dataset.

## How to run:
The original images needs to be resized to reduce the computational time, this is done with \texttt{resize_images.py}, the images can be reduced to desired shape.\n
\n
Before running the main file, the csv files needs to be modified. This is done using the \texttt{prepare_data.py} file. To run this, the pathway for train_csv, test_csv and duplicates_csv must be modified. Running this script will create the proessed train and test csv that is needed to run \texttt{main.py}. \n
\n
To run the main function, \texttt{main.py}, the paths: train_img_path and test_img_path should be changed to the path to the directory that contains the resized train and test images.
Also, the data_train and test_train should be change to the path to the proessed train and test csv files are located.\n
