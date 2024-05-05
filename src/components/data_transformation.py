import sys
from dataclasses import dataclass
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.compose import ColumnTransformer
import numpy as np 
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def replace_values(self, df, value_map):
        return df.replace(value_map)

    
    

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=HashingVectorizer()

            review_train_df=train_df['review']
            sentiment_train_df=train_df['sentiment']

            review_test_df=test_df['review']
            sentiment_test_df=test_df['sentiment']

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            review_train_vec=preprocessing_obj.transform(review_train_df)
            review_test_vec=preprocessing_obj.transform(review_test_df)
            sentiment_train = self.replace_values(sentiment_train_df, {'positive': 1, 'negative': 0})
            sentiment_test = self.replace_values(sentiment_test_df, {'positive': 1, 'negative': 0})
            sentiment_train_sparse= csr_matrix(np.array(sentiment_train).reshape(-1, 1))
            sentiment_test_sparse = csr_matrix(np.array(sentiment_test).reshape(-1, 1))
            # sentiment_train_sparse = sparse.csr_matrix(sentiment_train)
            # sentiment_test_sparse = sparse.csr_matrix(sentiment_test)


            train_vec =  hstack([
                review_train_vec, sentiment_train_sparse
            ])
            test_vec = hstack([
                review_test_vec, sentiment_test_sparse
            ])

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_vec,
                test_vec,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
