import os
import sys
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfing:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfing()
    
    
    def initiate_model_trainer(self, X_train_vector,y_train_vector, X_test_vector, y_test_vector):
         try:
             logging.info("Splitting training and test input data")
             X_train, y_train, X_test, y_test = (
                 X_train_vector,
                 y_train_vector,
                 X_test_vector,
                 y_test_vector
             )
             models = {
                "Logistic_Regression" : LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "SVC": SVC()
            }
             params = {
                "Logistic_Regression": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [50, 100, 200],
                    # 'max_depth': [3, 5, 7, 9]
                },
                "RandomForestClassifier": {
                    'n_estimators': [50, 100, 200],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    # 'max_depth': [None, 10, 20, 30],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4]
                },
                "AdaBoostClassifier": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [50, 100, 200]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    # 'max_depth': [None, 10, 20, 30],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4]
                },
                "SVC": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf', 'poly'],
                    # 'gamma': ['scale', 'auto']
                }
}
             model_report:dict=evaluate_models(X_train = X_train,y_train = y_train,X_test = X_test, y_test = y_test, models = models, param=params)
            ## To get best model score from dict
             best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

             best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
             best_model = models[best_model_name]

             if best_model_score<0.6:
                raise CustomException("No best model found")
             logging.info(f"Best found model on both training and testing dataset")

             save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

             predicted=best_model.predict(X_test)

             f1_score_ = f1_score(y_test, predicted)
             return f1_score_
         except Exception as e:
             raise CustomException(e,sys)
             
