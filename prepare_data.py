import sys
import pandas as pd
from preprocess import standardise_age, one_hot_encoding_anatomic_site, one_hot_encoding_sex

def prepare_data(train_csv, test_csv, duplicates_csv):
    """
    Prepares the data using the functions in preprocess.py.

    Args:
        train_csv: A string that gives the filepath to the original train_csv file described in 
            README.md.
        test_csv: A string that gives the filepath to the original test_csv file described in 
            README.md.
        duplicates_csv: A string that gives the filepath to the file that contains the duplicates in 
            the train_csv.
    
    Returns:
        Two new csv files is created: data/train_processed.csv and data/test_processed.csv
    """
    # Read the csv files and remove duplicates
    train_set = pd.read_csv(train_csv)
    test_set = pd.read_csv(test_csv)
    duplicates = pd.read_csv(duplicates_csv)
    train_set = train_set[~train_set['image_name'].isin(duplicates['ISIC_id'])]

    train_set = standardise_age(train_set)
    train_set['age_approx'] = train_set['age_approx'].fillna(0) # Fill NaNs with 0 (the mean)
    train_set = one_hot_encoding_anatomic_site(train_set)
    train_set = one_hot_encoding_sex(train_set)

    test_set = standardise_age(test_set)
    test_set = one_hot_encoding_anatomic_site(test_set)
    test_set = one_hot_encoding_sex(test_set)

    train_set.to_csv('data/train_processed.csv', index=False)
    test_set.to_csv('data/test_processed.csv', index=False)


if __name__ == '__main__':
    prepare_data(*sys.argv[1:])
 