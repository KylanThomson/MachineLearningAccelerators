import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer as si
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
- loads data from csv
- flags columns containing numerical values for handling missing values
- handles missing values
- encodes categorical data
- encodes labels
- splits data -> x_train, x_test, y_train, y_test
- (optional) Feature Scaling in the form of Standardisation or Normalisation

Note: normalization is recommended when you have a normal distribution, standardisation will always work for feature scaling
'''


def preprocessing(filename):
    
    # import .csv dataset as a numpy array object
    dataset = pd.read_csv(filename)
    
    # x is everything except the last column
    x = dataset.iloc[:, :-1].values
    start, end =  flag_number_columns(x)
    x[:, start:end] = handle_missing_data( x[:, start:end])
    x_len_before_encoding = x.shape[1]
    if start > 0:
      x = onehot_encoding(x, 0)
    x_len_after_encoding = x.shape[1]
    # x_dif is the index where numerical columns begin after the encoding is done
    x_dif = x_len_after_encoding - x_len_before_encoding + 1
    
    # y is dependent variables (last column by convention)
    y = dataset.iloc[:, -1].values
    # y = label_encoder(y)
    x_train, x_test, y_train, y_test = split_data(x,y)
    # x_train, x_test = feature_scaling(x_train, x_test, x_dif)
    return x_train, x_test, y_train, y_test

# Replaces missing values in a column with the mean for that column
def handle_missing_data(data):
    if data.shape[1] == 0:
      return data
    else:
      imputer = si(missing_values=np.nan, strategy='mean')
      imputer.fit(data[:])
      return imputer.transform(data[:])

# Represent categorical data with binary
def onehot_encoding(data, column):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),  [column])], remainder='passthrough')
    data = np.array(ct.fit_transform(data))
    return data

def label_encoder(data):
  le = LabelEncoder()
  return le.fit_transform(data)

# Splitting the into x_train, x_test, y_train, y_test
def split_data(x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
  return x_train, x_test, y_train, y_test

# feature scaling - a method used to normalize the range of independent variables or features of data. Only apply this function to columns that contain numerical values
def feature_scaling(x_train, x_test, x_dif):
  sc = StandardScaler()
  # todo: dynamically find where the numerical columns pickup
  x_train[:, x_dif:] = sc.fit_transform(x_train[:, x_dif:])
  x_test[:, x_dif:] = sc.transform(x_test[:, x_dif:])
  return x_train, x_test

def flag_number_columns(data):
  first_row = data[1, ...]
  flagged_columns = []
  if first_row.size == 1:
    return 0, 0
  else:
    for i in range(len(first_row)):
      if type(first_row[i]) == int or type(first_row[i]) == float:
        flagged_columns.append(i)
    return flagged_columns[0], flagged_columns[1]
  