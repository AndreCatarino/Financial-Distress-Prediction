# Financial-Distress-Prediction
Project Overview
- This project follows the CRISP-DM methodology, a recognized framework for data science workflows. Comprising 6 iterative phases, CRISP-DM guides the tasks essential for effective data mining.

Aim
- The project focuses on addressing a challenging multivariate time series classification problem using machine learning. The objective is to determine whether a company is financially distressed.

Data Intricacies
- Class Imbalance: The dataset presents a severe class imbalance, with only 3.7% of the observations being financially distressed. To address this, a dual strategy was applied involving oversampling and undersampling techniques. The Synthetic Minority Oversampling Technique (SMOTE) was employed to generate synthetic instances for the minority class, while random undersampling was implemented for the majority class, as suggested by the original paper on SMOTE. This approach ensures a balanced representation of both classes in the training data.
- Sequential Data: A custom forward chaining method was implemented to handle the sequential nature of time series data. This addresses the limitations of the Time Series Split function, which assumes that each row represents a data point from a unique time instance, with rows arranged in increasing order of time. However, this assumption does not hold for the dataset. In this sense, a similar cross-validation approach was designed to replace this function.

Evaluation Metrics
- Given the imbalanced nature of the dataset, accuracy metric may be misleading. Therefore, the F1 score, a metric that balances precision and recall, is chosen as the primary evaluation metric. Additionally, ROC curves and precision-recall curves are explored to gain a nuanced understanding of the model's performance under different probability thresholds. Threshold tuning for imbalanced classification is applied to enhance the model's effectiveness, particularly for minority classes.