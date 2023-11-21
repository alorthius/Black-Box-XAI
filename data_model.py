from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


class DataModel:

    def __init__(self, N: int) -> None:
        self.N = N
        self.X = self.generate_x()
        self.y = self.calculate_y(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.X, self.y)

    def generate_x(self) -> pd.DataFrame:
        X = None
        return X

    def calculate_y(self, X: pd.DataFrame) -> np.ndarray:
        y = None
        return y

    @staticmethod
    def split_data(X, y, seed: int = 1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def total_dependence(self, feature: str) -> (np.ndarray, np.ndarray, np.ndarray):
        grid_points = np.linspace(np.min(self.X[feature]), np.max(self.X[feature]), self.N)
        tdp = np.zeros(len(grid_points))

        for i, value in enumerate(grid_points):
            X_copy = self.X.copy()
            X_copy[feature] = value

            predictions = self._true_predictions(feature, X_copy)
            tdp[i] = np.mean(predictions)

        return grid_points, tdp

    def _true_predictions(self, feature: str, X: pd.DataFrame):
        return self.calculate_y(X)

    def __str__(self) -> str:
        return ""
