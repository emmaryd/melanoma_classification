import pandas as pd
import numpy as np

def one_hot_encoding_sex(df):
    """
    Creates two columns ('male'/'female') from one ('sex')
    """
    df['female'] = (df['sex']=='female').astype(int)
    df['male'] = (df['sex']=='male').astype(int)
    df.drop('sex', axis=1, inplace=True)
    return df

def normalize_age(df):
    """
    Normalizes the ages by dividing with the max age.
    """
    df['age_approx'] = (df['age_approx']- df['age_approx'].mean()) / (df['age_approx'].std()) 
    
    return df

def one_hot_encoding_anatomic_site(df):
    """
    Creates several new columns from the 'anatom_site_general_challenge' column,
    according to the one hot encodig principles.
    """
    # convert string to integer for all categories in 'anatom_site_general_challenge'

    df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna('NaN')

    for i in df.index:
        anatom_site = df.loc[i, 'anatom_site_general_challenge']
        if anatom_site in df.columns:
            df.loc[i, anatom_site] = 1
        else:
            df[anatom_site] = np.nan
            df.loc[i, anatom_site] = 1
            
    df = df.fillna(0)
    df.drop(['anatom_site_general_challenge', 'NaN'], axis=1, inplace=True)
    
    return df
