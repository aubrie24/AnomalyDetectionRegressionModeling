# assignment_4
Data Mining Assignment 4
The target column is "Inhibition"
We need to use regression because the Inhibition column is numerical and continuous
and regression models can output a predicted inhibition score for ranking molecules.

-need to:
  -clean data (check for outliers using anomaly detection, handle missing values)
  -split published data into dev and train sets. 
  -use train set for training and predict/evaluate on itself
  -use the dev set to predict and evaluate
  -create baseline 
  -create best model
  -pick the best model based on scores from train and dev
  -save the trained pipeline in output
  -predict using the new_molecules data and output the ranked data
