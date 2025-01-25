import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from joblib import dump
from joblib import load
import joblib

def predict(input_data):
    with open('best_pipeline.pkl', 'rb') as file:
        best_pipelines = pickle.load(file)
        dump(best_pipelines, 'best_pipeline.joblib')
    best_pipeline = load('best_pipeline.joblib')

    X_train_columns = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
                       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
                       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterQual',
                       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
                       'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                       'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
                       'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
                       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
                       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
                       'SaleCondition', 'TotalSF']

    important_fields = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalSF', 'YearBuilt']

    input_dict = {col: np.nan for col in X_train_columns}
    for col in important_fields:
        input_dict[col] = input_data.get(col, np.nan)

    input_df = pd.DataFrame([input_dict], columns=X_train_columns)

    print("Input to model:\n", input_df)  # Debug input
    predicted_log_price = best_pipeline.predict(input_df)
    predicted_price = np.expm1(predicted_log_price)

    print(f"Predicted House Price: ${predicted_price[0]:,.2f}")
    return predicted_price

input_data = {
            'OverallQual': float(1.34),
            'GrLivArea': float(3.003),
            'GarageCars': float(1),
            'TotalSF': float(2.87483),
            'YearBuilt': float(1999)
        }
print(predict(input_data))