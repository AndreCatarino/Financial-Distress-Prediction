# Financial-Distress-Prediction
- This project follows CRISP-DM methodology, short for Cross-Industry Standard Process for Data Mining, one of the most recognized data science workflow frameworks. CRISP-DM consists of 6 iterative phases that each have their designated tasks that ought to be fulfilled.
- The aim of this project is to tackle a  a multivariate time series classification problem with machine learning, to determine whether a company is financially distressed.
- The explored dataset intricacies include a severe class imabalance, wich was addressed by employing data augmentation for the minority class, using the Synthetic Minority Oversampling Technique (SMOTE). SMOTE is commonly used for oversampling imbalanced classification datasets. Other challenge relates to implementing forward chaining, suitable for sequential data such as time series. A cross-validation approach was designed to replace the use of Time Series Split. The Time Series Split function assumes that each row represents a data point from a unique instance of time, with rows arranged in increasing order of time. However, this assumption does not hold for our dataset.
