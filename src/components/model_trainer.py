import numpy as np
import pandas as pd
import os, sys
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Beginning with the model training process")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:,-1],
            )
            models = {
                "LineraRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
            }

            model_report: dict = evaluate_models(
                models, X_train, y_train, X_test, y_test
            )
            print("\n===============================================\n")
            logging.info(f"Model Report : {model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            print(
                f"Best model is {best_model_name} and the best R2 score is {best_model_score}"
            )
            logging.info(
                f"Best model is {best_model_name} and the best R2 score is {best_model_score}"
            )
            print("\n===============================================\n")

            best_model = models[best_model_name]
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

        except Exception as e:
            raise CustomException(e, sys)
