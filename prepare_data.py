import pandas as pd
from preprocess import normalize_age, one_hot_encoding_anatomic_site, one_hot_encoding_sex
import sys

def prepare_data():
    """
    Prepares the data.  Using the functions in preprocess.py.
    train_csv, test_csv should be file paths to the original csv files containing information 
    about the train and test sets respectively and duplicates_csv is a file containing the duplicates.
    Two new csv files will be created: data/train_processed.csv and data/test_processed.csv
    """
    train_csv = "/Users/emmarydholm/Documents/code/melanoma_classification/melanoma_classification/data/train.csv"
    test_csv =  "/Users/emmarydholm/Documents/code/melanoma_classification/melanoma_classification/data/test.csv"
    duplicates_csv = "/Users/emmarydholm/Documents/code/melanoma_classification/melanoma_classification/data/duplicates.csv"
    train_set = pd.read_csv(train_csv)
    test_set = pd.read_csv(test_csv)

    # the train set contains some duplicates these are to be removed before training
    duplicates = pd.read_csv(duplicates_csv)
    train_set = train_set[~train_set['image_name'].isin(duplicates['ISIC_id'])]

    train_set = normalize_age(train_set)
    train_set['age_approx'] = train_set['age_approx'].fillna(0) # Fill NaNs with 0 ( the mean after normalization )
    train_set = one_hot_encoding_anatomic_site(train_set)
    train_set = one_hot_encoding_sex(train_set)

    test_set = normalize_age(test_set)
    test_set = one_hot_encoding_anatomic_site(test_set)
    test_set = one_hot_encoding_sex(test_set)

    train_set.to_csv('data/train_processed.csv', index=False)
    test_set.to_csv('data/test_processed.csv', index=False)


if __name__ == '__main__':
    prepare_data()
 