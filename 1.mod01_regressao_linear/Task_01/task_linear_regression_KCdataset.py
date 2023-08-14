import pandas as pd
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', 15)

data = pd.read_csv('./kc_house_data.csv')
# print(data.head())