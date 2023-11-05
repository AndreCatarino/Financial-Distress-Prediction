from sklearn.metrics import accuracy_score
import pandas as pd
from typing import List

""" 
Approach to forward chaining cross validation:
    - Split the dataset into multiple groups: 1 group per Company
    - For each group, derive the indices for Forward Chaining
    - Combine the list of indices per group into 1 final index list

In this sense, 'TimeSeriesCrossValidator' class is designed for forward chaining time series cross validation.
It distributes the indices into 'n_splits' parts, taking one part of the indices from each group and combining them,for each split.
This distribution of indices ensures that data from each group (company) is included in both train and test sets in a sequential manner.
"""

class TimeSeriesCrossValidator:
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, n_splits:int=5) -> None:
        """
        Constructor for TimeSeriesCrossValidator class
        :param X: Feature matrix
        :param y: Target vector
        :param n_splits: Number of splits
        :return: None
        """
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.groups = self.X.groupby('Company').groups
        self.group_indexes = self._get_group_indexes()
        self.indexes = self._get_indexes()

    def _get_group_indexes(self) -> List[List[int]]:
        """
        Obtain the indexes for each group
        :return: List of indexes for each group
        """
        group_indexes = []
        for group in self.groups:
            group_indexes.append(self.groups[group].tolist())
        return group_indexes

    def _get_indexes(self) -> List[List[int]]:
        """
        Generates a list of indexes for each split
        :return: List of indexes for each split
        """
        indexes = []
        for i in range(self.n_splits):
            indexes.append([])
        # For each group, distributes the indices into n_splits in a way that follows Forward Chaining
        for group_index in self.group_indexes:
            for i in range(self.n_splits):
                indexes[i].extend(group_index[i::self.n_splits])
        return indexes

    def split(self) -> tuple:
        """
        Split the data into train and test sets
        :return: Train and test sets
        """
        for i in range(self.n_splits):
            train_index = self.indexes[i]
            test_index = self.indexes[i+1]
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            yield X_train, X_test, y_train, y_test

    def evaluate(self, model:object, scoring:str='accuracy') -> list:
        """
        Evaluate the model using cross validation
        :param model: Model to be evaluated
        :param scoring: Scoring metric
        :return: List of scores
        """
        scores = []
        for X_train, X_test, y_train, y_test in self.split():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        return scores
