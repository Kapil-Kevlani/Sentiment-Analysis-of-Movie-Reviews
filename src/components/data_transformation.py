import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    vectorizer_obj_file_path = os.path.join('artifacts', "tfidf_vectorizer.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            text_column = "review"
            target_column = "sentiment"

            text_pipeline = Pipeline(
                steps=[
                    ("vectorizer", TfidfVectorizer(stop_words='english'))
                ]
            )

            logging.info(f"Text column: {text_column}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("text_pipeline", text_pipeline, text_column)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Initialize preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Separate features and labels
            input_feature_train_df = train_df.drop(columns=['sentiment'], axis=1)
            target_feature_train_df = train_df['sentiment']

            input_feature_test_df = test_df.drop(columns=['sentiment'], axis=1)
            target_feature_test_df = test_df['sentiment']

            # Transform features using the preprocessing pipeline
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Encode labels using LabelEncoder
            label_encoder = LabelEncoder()
            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            # Optionally save the preprocessor and label encoder
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)
            # save_object(self.data_transformation_config.vectorizer_obj_file_path, label_encoder)

            # Return processed features and labels (kept separate)
            return (
                input_feature_train_arr,
                target_feature_train_arr,
                input_feature_test_arr,
                target_feature_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                # self.data_transformation_config.vectorizer_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)