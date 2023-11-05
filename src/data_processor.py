from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from numpy import where
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class DataProcessor:
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Constructor for DataProcessor class
        :param data: Dataframe to be processed
        :return: None
        """
        self.data = data
        self.X = None
        self.y = None

    def create_feature_matrix_and_target_vector(self, target_column: str) -> tuple:
        """
        Create feature matrix and target vector
        :param target_column: Name of the target column
        :return: Feature matrix and target vector
        """
        self.X = self.data.drop(target_column, axis=1)
        self.y = self.data[target_column]
        return self.X, self.y

    def scale(self) -> pd.DataFrame:
        """ 
        Scale the data using StandardScaler
        :return: Scaled features dataframe
        """
        if self.X is None:
            raise Exception("Feature matrix not found. Run create_feature_matrix_and_target_vector() first.")
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(self.X)
        return pd.DataFrame(scaled_X, columns=self.X.columns)
    
    def class_distribution(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        """
        Plot the class distribution
        :param X: Feature matrix
        :param y: Target vector
        :return: None
        """
        # summarize class distribution
        counter = Counter(y)
        print(counter)
        # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            plt.scatter(X.iloc[row_ix, 0], X.iloc[row_ix, 3], label=str(label))
        plt.legend()
        plt.show()
    
    def transform(self, X:pd.DataFrame, y:pd.DataFrame, over_ratio:float, under_ratio:float) -> tuple:
        """
        Transform the data using SMOTE and RandomUnderSampler
        :param X: Feature matrix
        :param y: Target vector
        :param over_ratio: Ratio for SMOTE
        :param under_ratio: Ratio for RandomUnderSampler
        :return: Transformed feature matrix and target vector
        """
        # define pipeline
        oversampling = SMOTE(sampling_strategy=over_ratio)
        undersampling = RandomUnderSampler(sampling_strategy=under_ratio)
        steps = [('o', oversampling), ('u', undersampling)]
        pipeline = Pipeline(steps=steps)
        # transform the dataset
        X, y = pipeline.fit_resample(X, y)
        # summarize the new class distribution
        counter = Counter(y)
        print(counter)
        # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            plt.scatter(X.iloc[row_ix, 0], X.iloc[row_ix, 3], label=str(label))
        plt.legend()
        plt.show()
        return X, y
        