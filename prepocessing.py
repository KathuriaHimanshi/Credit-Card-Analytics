# Feature Analysis & Data Preprocessing 

# Importing the libraries
import numpy as np    #library that contains mathematical tools.
import matplotlib.pyplot as plt   #going to help us plot nice charts.
import pandas as pd #best library to import data sets and manage data sets.

# Importing the dataset
new_puzzle_train_dataset = pd.read_csv('new_puzzle_train_dataset.csv')
puzzle_test_dataset = pd.read_csv('puzzle_test_dataset.csv')

# Feature Analysis
# Find number of columns and rows in both training and testing data

train_shape=new_puzzle_train_dataset.shape
test_shape=puzzle_test_dataset.shape

# Describing Train Data 
train_description = new_puzzle_train_dataset.describe()

# Describing Test Data 
test_description = puzzle_test_dataset.describe()


# Count Null Values in each column
# For training set
null_training = new_puzzle_train_dataset.isnull().sum(axis = 0).reset_index()
 
# For training set
null_test = puzzle_test_dataset.isnull().sum(axis = 0).reset_index()

# Taking the relavant features after dropping those mentioned in PPT plus 'ids' to be used as Index
train_dataset = new_puzzle_train_dataset[['ids','default','score_3','score_4', 'score_5', 'score_6',
                                          'risk_rate','amount_borrowed','borrowed_in_months', 'credit_limit',
                                          'income', 'gender', 'facebook_profile', 'n_bankruptcies', 
                                          'n_defaulted_loans' , 'n_accounts', 'n_issues' ,
                                          'profit']]
test_dataset = puzzle_test_dataset[['ids','score_3','score_4', 'score_5', 'score_6',
                                    'risk_rate','amount_borrowed','borrowed_in_months', 'credit_limit',
                                    'income', 'gender', 'facebook_profile', 'n_bankruptcies', 
                                    'n_defaulted_loans' , 'n_accounts', 'n_issues']]

# Spliting X_train and y_train
"""
# counting number of females and males
train_dataset.groupby('gender').size() 
test_dataset.groupby('gender').size()
# more males are there both in test and train data . Therefore, assuming missing values also to be male

"""

# Fill empty values of column 'gender'
train_dataset['gender'] = train_dataset['gender'] .fillna('m')
test_dataset['gender'] = test_dataset['gender'] .fillna('m')
# Label encoding
train_dataset['gender'] = train_dataset['gender'].map( {"f": 1, "m": 0} ).astype(int)
test_dataset['gender'] = test_dataset['gender'].map( {'f': 1, 'm': 0} ).astype(int)

"""
# counting number of TRUE/False vales
train_dataset.groupby('facebook_profile').size() 
test_dataset.groupby('facebook_profile').size()

"""

# Fill empty values of column 'facebook_profile' and encoding them as 0/1
facebook_profile_mapping = {"TRUE": 1, "FALSE": 0}
        train_dataset['facebook_profile'] = train_dataset['facebook_profile'].map(facebook_profile_mapping)
        train_dataset['facebook_profile'] = train_dataset['facebook_profile'].fillna(0)
        test_dataset['facebook_profile'] = test_dataset['facebook_profile'].map(facebook_profile_mapping)
        test_dataset['facebook_profile'] = test_dataset['facebook_profile'].fillna(0)

# Find number of missing values in each column
train_dataset.info()
test_dataset.info()

# Taking care of missing data - all numerical continious variables . Therefore, using mean values
# Training data
train_dataset['score_3'].fillna(train_dataset['score_3'].dropna().mean(), inplace=True)
train_dataset['risk_rate'].fillna(train_dataset['risk_rate'].dropna().mean(), inplace=True)
train_dataset['amount_borrowed'].fillna(train_dataset['amount_borrowed'].dropna().mean(), inplace=True)
train_dataset['borrowed_in_months'].fillna(train_dataset['borrowed_in_months'].dropna().mean(), inplace=True)
train_dataset['credit_limit'].fillna(train_dataset['credit_limit'].dropna().mean(), inplace=True)
train_dataset['income'].fillna(train_dataset['income'].dropna().mean(), inplace=True)
train_dataset['n_bankruptcies'].fillna(train_dataset['n_bankruptcies'].dropna().mean(), inplace=True)
train_dataset['n_defaulted_loans'].fillna(train_dataset['n_defaulted_loans'].dropna().mean(), inplace=True)
train_dataset['n_accounts'].fillna(train_dataset['n_accounts'].dropna().mean(), inplace=True)
train_dataset['n_issues']=train_dataset['n_issues'].fillna(0)

# Testing Data
test_dataset['score_3'].fillna(train_dataset['score_3'].dropna().mean(), inplace=True)
test_dataset['risk_rate'].fillna(train_dataset['risk_rate'].dropna().mean(), inplace=True)
test_dataset['amount_borrowed'].fillna(train_dataset['amount_borrowed'].dropna().mean(), inplace=True)
test_dataset['borrowed_in_months'].fillna(train_dataset['borrowed_in_months'].dropna().mean(), inplace=True)
test_dataset['credit_limit'].fillna(train_dataset['credit_limit'].dropna().mean(), inplace=True)
test_dataset['income'].fillna(train_dataset['income'].dropna().mean(), inplace=True)
test_dataset['n_bankruptcies'].fillna(train_dataset['n_bankruptcies'].dropna().mean(), inplace=True)
test_dataset['n_defaulted_loans'].fillna(train_dataset['n_defaulted_loans'].dropna().mean(), inplace=True)
test_dataset['n_accounts'].fillna(train_dataset['n_accounts'].dropna().mean(), inplace=True)
test_dataset['n_issues']=train_dataset['n_issues'].fillna(0)

"""
coeff = pd.DataFrame(train_dataset.columns.delete(0))
coeff.columns = ['Feature']
coeff["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

"""

# Saving dataset to csv files
train_dataset.to_csv('train_dataset.csv')
test_dataset.to_csv('test_dataset.csv')



