import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_vectorized = preprocessor.transform(features)
            preds = model.predict(data_vectorized)
            return preds
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self, review):
        self.review = review
    

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "review" : [self.review]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
            
