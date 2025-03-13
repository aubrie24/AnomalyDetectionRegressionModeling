# assignment_4
Data Mining Assignment 4
The target column is "Inhibition"
We need to use regression because the Inhibition column is numerical and continuous
and regression models can output a predicted inhibition score for ranking molecules.

-tasks left to do:
  -clean data (check for outliers using anomaly detection, etc)
  -create custom transformers
  -create best pipeline (the baseline is at .1 MSE)
  -pick the best model based on scores from train and dev
  -save the trained pipeline in output
  -predict using the new_molecules data and output the ranked data
