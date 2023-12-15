from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams["grid.linestyle"] = "--"


class DataModel:

    def __init__(self, N: int) -> None:
        self.N = N
        self.X = self.generate_x()
        self.y = self.calculate_y(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.X, self.y)

    def generate_x(self) -> pd.DataFrame:
        # Overload this
        X = None
        return X

    def calculate_y(self, X: pd.DataFrame) -> np.ndarray:
        # Overload this
        y = None
        return y

    def get_feature_names(self) -> list[str]:
        return self.X.columns.tolist()

    @staticmethod
    def split_data(X, y, seed: int = 1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def true_dependence(self, feature: str) -> (np.ndarray, np.ndarray, np.ndarray):
        grid_points = np.linspace(np.min(self.X[feature]), np.max(self.X[feature]), self.N)
        tdp = np.zeros(len(grid_points))

        for i, value in enumerate(grid_points):
            X_copy = self.X.copy()
            X_copy[feature] = value

            predictions = self.true_predictions(feature, X_copy)
            tdp[i] = np.mean(predictions)

        return grid_points, tdp

    def total_dependence(self, feature: str, ml_model) -> (np.ndarray, np.ndarray, np.ndarray):
        grid_points = np.linspace(np.min(self.X[feature]), np.max(self.X[feature]), self.N)
        tdp = np.zeros(len(grid_points))

        for i, value in enumerate(grid_points):
            X_copy = self.X.copy()
            X_copy[feature] = value

            predictions = ml_model.predict(X_copy)
            tdp[i] = np.mean(predictions)

        return grid_points, tdp

    def true_predictions(self, feature: str, X: pd.DataFrame):
        # Overload this
        return self.calculate_y(X)

    def plot_true_dp(self, feature: str, to_show: bool = True, alpha: float = 1):
        grid, tdp = self.true_dependence(feature)
        plt.plot(grid, tdp, color="blue", lw=1.5, ls="-.", alpha=alpha, label="True Dependence")
        plt.ylim([np.min(self.y), np.max(self.y)])
        plt.xlabel(f"{feature}")
        plt.ylabel("$y$")
        if to_show:
            plt.show()

    def plot_total_dp(self, feature: str, ml_model, to_show: bool = True, alpha: float = 1):
        grid, tdp = self.total_dependence(feature, ml_model)
        plt.plot(grid, tdp, color="green", lw=1.5, ls="-.", alpha=alpha, label="Total Dependence")
        plt.ylim([np.min(self.y), np.max(self.y)])
        plt.xlabel(f"{feature}")
        plt.ylabel("$\hat y$")
        if to_show:
            plt.show()

    def plot_true_and_total_dp(self, feature: str, ml_model):
        self.plot_total_dp(feature, ml_model, to_show=False)
        self.plot_true_dp(feature, to_show=False, alpha=0.6)

        plt.title(feature)
        plt.legend()
        plt.show()

    def __str__(self) -> str:
        # Overload this
        return ""
