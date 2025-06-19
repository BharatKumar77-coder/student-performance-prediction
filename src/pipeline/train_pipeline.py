import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_path = os.path.join('notebook', 'data', 'stud.csv')  
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            logging.info("Data loaded successfully.")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def build_preprocessor(self):
        try:
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_features = ['reading_score', 'writing_score']

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        try:
            df = self.load_data()

            # Split features and target
            X = df.drop(columns=["math_score"])
            y = df["math_score"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            preprocessor = self.build_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save preprocessor
            save_object(self.preprocessor_path, preprocessor)
            logging.info("Preprocessor saved.")

            # Combine into arrays to pass to model trainer
            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            # Train model
            trainer = ModelTrainer()
            r2_score_result = trainer.initiate_model_trainer(train_arr, test_arr)
            print(f"Model training completed. R2 Score: {r2_score_result}")
        except Exception as e:
            raise CustomException(e, sys)

