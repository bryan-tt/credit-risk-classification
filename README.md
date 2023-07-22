# credit-risk-classification
## Overview of the Analysis
The purpose of this analysis is to train and evaluate a supervised machine learning model based on loan risk. The dataset includes historical lending activity from a peer-to-peer lending services company. The goal is to build a model that can predict the creditworthiness of borrowers.

### About the Data
The dataset includes 77,536 records, and 8 columns.
* The features include:
    - <strong>loan_size</strong> (total loan amount)
    - <strong>interest_rate</strong> (interest rate for the loan)
    - <strong>borrower_income</strong> (total income of indivdual)
    - <strong>debt_to_income</strong> (debt to income ratio)
    - <strong>num_of_accounts</strong> (total number of credit accounts)
    - <strong>derogatory_marks</strong> (negative marks including late payments, defaults, charge-offs, collections, bankruptcies, foreclosures, tax liens, etc.)
    - <strong>total_debt</strong> (total debt of individual)
* The target value is:
    - <strong>loan_status</strong> (`0` (healthy loan) and `1` (high-risk loan))
### Logistic Regression Machine Learning Model Process
1. With this data, I decided not to scale it since we are using a Logistic Regression model, which can handle unscaled data.
2. Separate the features from the target
3. Split the data for training and testing
4. Create the model with sklearn `LogisticRegression`
5. Fit the model using training data
6. Make the prediction with the model on the testing data
7. Calculate the evaluation metrics: accuracy, precision, recall, confusion matrix, classification report
8. Resample the training data with `RandomOverSampler` to address the issue of imbalanced data & repeat steps 1-7 for the resampled data model.

## Results
* Machine Learning Model 1: Logistic Regression Model
  * Accuracy: 99.2%
  * Balanced Accuracy: 94.4%
  * Precision
    * Healthy loan: 100%
    * High-risk loan: 87%
  * Recall scores
    * Healthy loan: 100%
    * High-risk loan: 89%

* Machine Learning Model 2: Oversampled Logistic Regression Model
  * Accuracy: 99.5%
  * Balanced Accuracy: 99.6%
  * Precision
    * Healthy loan: 100%
    * High-risk loan: 87%
  * Recall scores
    * Healthy loan: 100%
    * High-risk loan: 100%

## Summary

In terms of evaluation metrics, it looks like the oversampled model has a slightly better overall accuracy, and a better recall score for the high-risk loans, making it the better performing model. It's more important to predict high-risk loans since the high-risk loans are the ones that will do more harm to the companies that service the loans. Since the recall score for the high-risk loans was improved in the 2nd model, this signifies that model can effectively capture more high-risk loan instances.
