import pandas as pd
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
        # For each group, distributes the indices into n_splits 
        for group_index in self.group_indexes:
            for i in range(self.n_splits):
                indexes[i].extend(group_index[i::self.n_splits])
        return indexes

    def split(self) -> tuple:
        """
        For each split, yield the train and test sets using the indexes obtained from _get_indexes() respecting the forward chaining approach
        :return: Train and test sets
        """
        for i in range(self.n_splits):
            train_indexes = []
            test_indexes = self.indexes[i]  # Use the current fold's indexes for testing
            for j in range(i):
                train_indexes.extend(self.indexes[j])  # Include all previous folds in training data
            yield self.X.iloc[train_indexes], self.X.iloc[test_indexes], self.y.iloc[train_indexes], self.y.iloc[test_indexes]

    def evaluate(self, model:object, return_model:bool=False) -> tuple:
        """
        Evaluate the model using cross validation
        :param model: Model to be evaluated
        :param scoring: Scoring metric
        :return: List of scores
        """
        acc_scores = []
        prec_scores = []
        rec_scores = []
        f1_scores = []
        roc_auc_scores = []
        for X_train, X_test, y_train, y_test in self.split():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_scores.append(accuracy_score(y_test, y_pred))
            prec_scores.append(precision_score(y_test, y_pred))
            rec_scores.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            roc_auc_scores.append(roc_auc_score(y_test, y_pred))
        if return_model:
            return model, acc_scores, prec_scores, rec_scores, f1_scores, roc_auc_scores
        return acc_scores, prec_scores, rec_scores, f1_scores, roc_auc_scores
