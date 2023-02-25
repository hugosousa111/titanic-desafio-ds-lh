"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""

import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

def fill_nan_values(df: pd.DataFrame, features):
    """Fill NaN values

    Args:
        df (pd.DataFrame): dataframe
        features (List): model features 

    Returns:
        pd.DataFrame: dataframe without NaN values
    """
    all_feat_values_nan = df.isnull().sum()>0
    feat_with_nan = all_feat_values_nan.index[all_feat_values_nan == True]    
    features_inter = [x for x in feat_with_nan if x in features]
    for feat in features_inter: 
        if df[feat].dtype == float:
            df[feat] = df[feat].fillna(df[feat].mean())
        elif df[feat].dtype == object:
            df[feat] = df[feat].fillna(df[feat].mode()[0])
    
    return df

def feature_engineering_dataset_test(
    df: pd.DataFrame, 
    df_test:pd.DataFrame,
    features, 
    oe,
    categorical_variables
    ):
    """Feature engineering to test dataset,
    1. fill_nan_values
    2. ordinal_encoder_categorical_variables
    3. feature_and_target_selection
    4. create_feature_category
    
    Args:
        df (pd.DataFrame): train dataframe
        df_test (pd.DataFrame): test dataframe 
        features (List): model features 
        oe (OrdinalEncoder): trained ordinal encoder
        categorical_variables (List): categorical variables

    Returns:
        pd.DataFrame: clean test dataframe 
    """
    # fill_nan_values
    all_feat_values_nan = df_test.isnull().sum()>0
    feat_with_nan = all_feat_values_nan.index[all_feat_values_nan == True]    
    features_inter = [x for x in feat_with_nan if x in features]
    for feat in features_inter: 
        if feat == 'Age':
            df_test[feat] = df_test[feat].fillna(df[feat].mean())
        elif feat == 'Fare':
            df_test[feat] = df_test[feat].fillna(df[feat].mode()[0])
    
    # ordinal_encoder_categorical_variables
    result = oe.transform(df_test[categorical_variables])
    df_test[categorical_variables] = result
    
    #feature_and_target_selection
    df_test = df_test[features]
    
    #create_feature_category
    df_test = create_feature_category(df_test, features, None)
    
    return df_test

def create_feature_category(df: pd.DataFrame, features, target): 
    """Creates categories for continuous attributes
    Age -> age_category
    Fare -> fare_category

    Args:
        df (pd.DataFrame): dataframe
        features (List): model features
        target (str): target string

    Returns:
        pd.DataFrame: dataframe with categories
    """
    #0 (Children) Age < 12
    #1 (Adolescents) 12 <= Age < 18
    #2 (Adults 1) 18 <= Age < 34
    #3 (Adults 2) 34 <= Age < 50
    #4 (Adults 3) 50 <= Age < 65
    #5 (Seniors) 65 <= Age
    df['age_category'] = 0
    df.loc[(df['Age'] < 12),'age_category'] = 0
    df.loc[(df['Age'] >= 12)&(df['Age'] < 18),'age_category'] = 1
    df.loc[(df['Age'] >= 18)&(df['Age'] < 34),'age_category'] = 2
    df.loc[(df['Age'] >= 34)&(df['Age'] < 50),'age_category'] = 3
    df.loc[(df['Age'] >= 50)&(df['Age'] < 65),'age_category'] = 4
    df.loc[(df['Age'] >= 65),'age_category'] = 5
    
    #0 Fare < 8
    #1 8 <= Fare < 13
    #2 13 <= Fare < 21
    #3 21 <= Fare < 27
    #4 27 <= Fare < 32
    #5 32 <= Fare < 84
    #6 84 <= Fare
    df['fare_category'] = 0
    df.loc[(df['Fare'] < 8),'fare_category'] = 0
    df.loc[(df['Fare'] >= 8)&(df['Fare'] < 13),'fare_category'] = 1
    df.loc[(df['Fare'] >= 13)&(df['Fare'] < 21),'fare_category'] = 2
    df.loc[(df['Fare'] >= 21)&(df['Fare'] < 27),'fare_category'] = 3
    df.loc[(df['Fare'] >= 27)&(df['Fare'] < 32),'fare_category'] = 4
    df.loc[(df['Fare'] >= 32)&(df['Fare'] < 84),'fare_category'] = 5
    df.loc[(df['Fare'] >= 84),'fare_category'] = 6

    if target == None: 
        df = df[features + ['age_category', 'fare_category']]
    else: 
        df = df[features + ['age_category', 'fare_category'] + [target]]
        
    df = df.drop(['Age', 'Fare'],axis=1)
    
    return df
    
def ordinal_encoder_categorical_variables(
    df: pd.DataFrame, categorical_variables: List
) -> Tuple[pd.DataFrame, OrdinalEncoder]:
    """Node transforms df categorical variables into ordinals

    Args:
        df (pd.DataFrame): dataframe
        categorical_variables (List): list of categorical var to transform

    Returns:
        Dict[pd.DataFrame, OrdinalEncoder]: return transformed df and the encoder
    """

    df_categorical_variables_transformed = df.copy()
    ordinal_encoder_transformer = OrdinalEncoder(dtype="int")

    result = ordinal_encoder_transformer.fit_transform(
        df_categorical_variables_transformed[categorical_variables]
    )

    df_categorical_variables_transformed[categorical_variables] = result

    return df_categorical_variables_transformed, ordinal_encoder_transformer

def feature_and_target_selection(df: pd.DataFrame, features, target):
    """ Node that selects only features and target 
    columns from the original dataframe

    Args:
        df (pd.DataFrame): dataframe
        features (List): model features
        target (str): target value

    Returns:
        pd.DataFrame: df with selected columns
    """

    df_with_selected_columns = df[features + [target]]

    return df_with_selected_columns

def data_split(df: pd.DataFrame, split_proportion: float = 0.1, random_state: int = None):
    """ Node wrapper for sklearn's train_test_split. Splits the original dataset into two. 
    
    This node is expected to be leveraged twice on pipeline runs:
    The first split will reserve (by default) 10% of the dataset to be acessed exclusively on CI runs. In order for this to happen, 
    random_state is fixed so that all models will be validated against the same dataset (kaggle inspired). 
    Second data usage is the typical train/test split.

    Args:
        df: Dataframe to be split
        split_proportion: Percentage of the original dataset to be used as the validation/test dataset
        random_state: Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls 
    
    Returns:
        pd.DataFrame: train dataset
        pd.DataFrame: test dataset
    """ 

    train, test = train_test_split(
        df,
        test_size=split_proportion,
        random_state=random_state
    )

    return train, test
