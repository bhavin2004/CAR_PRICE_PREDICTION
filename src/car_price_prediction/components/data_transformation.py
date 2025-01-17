from src.car_price_prediction.utlis import save_pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

# from src.utlis import get_name_data,get_cast_names,fetch_director

# nltk.download('punkt')
# nltk.download('stopwords')

@dataclass
class DataTransformationConfig():
    preprocessor_pkl_path = Path('artifacts/preprocessor.pkl')
    
class DataTransformation():
    
    def __init__(self):
        self.config = DataTransformationConfig()
        
    def get_preprocessor(self,df:pd.DataFrame()):
        numerical_col = df.select_dtypes(exclude='object').columns
        categorical_col = df.select_dtypes(include='object').columns
        oh = OneHotEncoder()
        oh.fit(df[['name', 'company', 'fuel_type']])
        #creating preprocessor
        preprocessor = ColumnTransformer([
            ("numerical_cols",StandardScaler(),numerical_col),
            ('categorical_col',OneHotEncoder(categories=oh.categories_),categorical_col)
            ])
        
        return preprocessor
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Data Transformation is Initiated")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
    
            #Cleaning year column
            train_df = train_df[train_df['year'].str.isnumeric()]
            test_df = test_df[test_df['year'].str.isnumeric()]
            
            train_df['year']=train_df['year'].astype(np.int64)
            test_df['year']=test_df['year'].astype(np.int64)

            #Cleaning Price column
            train_df = train_df[train_df['Price']!='Ask For Price']
            test_df = test_df[test_df['Price']!='Ask For Price']
            train_df['Price']=train_df['Price'].str.replace(',','').astype(np.int64)
            test_df['Price']=test_df['Price'].str.replace(',','').astype(np.int64)
            
            #Cleaning km_driven column
            train_df['kms_driven']=train_df['kms_driven'].str.split(' ').str.get(0).str.replace(',','') 
            train_df=train_df[train_df['kms_driven'].str.isnumeric()]
            train_df['kms_driven'] =train_df['kms_driven'].astype(np.int64)
            
            test_df['kms_driven']=test_df['kms_driven'].str.split(' ').str.get(0).str.replace(',','') 
            test_df=test_df[test_df['kms_driven'].str.isnumeric()]
            test_df['kms_driven'] =test_df['kms_driven'].astype(np.int64)
            
            #cleaning fuel_type column
            train_df =train_df[train_df['fuel_type'].notna()]
            test_df =test_df[test_df['fuel_type'].notna()]
            
            #Taking first three word in name column
            train_df['name']=train_df['name'].apply(lambda x: " ".join(x.split(" ")[0:3]))
            test_df['name']=test_df['name'].apply(lambda x: " ".join(x.split(" ")[0:3]))

            #creating tmp df for preproccessor
            tmp_df=pd.concat([train_df,test_df],axis=0)
            tmp_df.drop(columns='Price',inplace=True)   
            tmp_df.to_csv('artifacts/cleaned_data.csv',header=True,index=False)  
            preprocessor = self.get_preprocessor(tmp_df)
            logging.info("Preprocessor loaded")
            
            logging.info(train_df.head().to_string())
            #spiliting the independent and dependent cols
            x_train = train_df.drop(columns='Price')
            x_test = test_df.drop(columns='Price')
            y_train = train_df['Price']
            y_test = test_df['Price']
            test_df.to_csv('artifacts/x_test.csv',header=True,index=False)

            # print(x_train.head())
            x_train = preprocessor.fit_transform(x_train).toarray()
            x_test = preprocessor.transform(x_test).toarray()
            
            logging.info("Preprocssed the data")
            
            #saving pkl file of preproccessor
            save_pkl(obj=preprocessor,obj_path=self.config.preprocessor_pkl_path)

            logging.info('Successfully saved Vectorizer pkl')
            logging.info('Data Transformation is Complete')
            return(
                x_train,
                x_test,
                y_train,
                y_test
            )     

        except Exception as e:
            logging.error(f"Error occurred in data transformation due to {e}")
            raise CustomException(e,sys)
        
