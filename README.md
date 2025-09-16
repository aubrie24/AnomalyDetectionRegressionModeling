Anomaly Detection & Regression Pipeline for Chemical Screening Data

1) Built an end to end pipeline for data cleaning, anomaly detection, and regression modeling.
2) Handles noisy chemical screening datasets by removing duplicates, managing missing values, and detecting anomalies.
3) Trains regression models (Linear Regression, Random Forest, Gradient Boosting) to predict inhibition scores.
4) Identifies significant molecular features driving predictions.

Features: 
1) Data Quality Reports: Missing values, duplicates, leakage risks, column stats.
2) Custom preprocessing: Numeric conversion, missing value handling, infinite value capping.
3) Anomaly Detection: Isolation Forest to flag/remove outliers.
4) Regression Modeling: Model training and selection with MSE evaluation.
5) Unseen Data Predictions: Ranked predictions with anomalies flagged as NaN.
6) Feature Significance Analysis: Statistical testing on top vs bottom ranked molecules.

Outputs:
1) Text based quality reports
2) Saved pipeline (.pkl)
3) Ranked predictions (.csv)
4) Significant features report (significant_features.csv)

*DISCLAIMER* The data used for this project was made available through Wake Forest University and is not published in this repository. Therefore, the code and output is for refrence and cannot be run. 

How to run regression.py
1) Install dependencies 
  pip install pandas numpy scikit-learn scipy joblib
2) python regression.py or python regression.py --test (for test trained model on new molecules)

How to run quality_report.py 
1) Install dependencies
   pip install pandas scikit-learn
2) python quality_report.py <>



  
   
