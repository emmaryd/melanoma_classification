import numpy as np

def one_hot_encoding_sex(df):
    """
    Creates two columns ('male'/'female') from one ('sex').
    
    Args:
        df: The dataframe, (train_cv or test_csv), needs to have one column called 'sex'.
    
    Returns:
        A modified version of the dataframe df.
    """
    df['female'] = (df['sex'] == 'female').astype(int)
    df['male'] = (df['sex'] == 'male').astype(int)
    df.drop('sex', axis=1, inplace=True)
    return df

def standardise_age(df):
    """ Standardise the ages by subtracting the mean and dividing with the standard diviation.

    Args:
        df: The dataframe, (train_cv or test_csv), needs to have one column called 'age_approx'.

    Returns:
        A modified version of the dataframe df.
    """

    df['age_approx'] = (df['age_approx']- df['age_approx'].mean()) / (df['age_approx'].std())
    
    return df

def one_hot_encoding_anatomic_site(df):
    """ Creates new columns from the 'anatom_site_general_challenge', using one hot encodig.
    Args:
        df: The dataframe, (train_cv or test_csv), needs to have one column called
            'anatom_site_general_challenge'.
    
    Returns:
        A modified version of the dataframe df.
    """

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
