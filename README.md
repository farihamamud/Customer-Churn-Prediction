# Customer Churn Prediction

## Problem
Churn means a customer cancels their subscription and leaves the company.
This project uses data from 7,043 telecom customers to predict which ones are likely to churn.

## Data
The dataset is the Telco Customer Churn data (IBM sample data, downloaded from Kaggle).
It has 7,043 customers and 21 columns: customer details, the services they use, billing information, and tenure (the number of months they have been a customer).
The target column is Churn. 1,869 customers (26.5%) churned and 5,174 stayed, so the classes are imbalanced.

## Cleaning
The TotalCharges column was stored as text, so I converted it to numbers.
This created 11 missing values.
All of these customers had tenure = 0, which means they had just joined and were not billed yet.
I filled these values with 0 because they had paid nothing so far.
I also changed Churn from No/Yes to 0/1 and removed customerID, because it is only a label.

## Findings
Customers who left had an average tenure of 18.0 months. Customers who stayed had 37.6 months.
Month-to-month contracts had a churn rate of 42.7%, one-year contracts 11.3%, and two-year contracts 2.8%.
Fiber optic customers churned more (41.9%) than DSL customers (19.0%).
Customers paying by electronic check churned the most (45.3%), compared with 15-17% for automatic payments.
These are patterns in the data. I cannot say they are the cause of churn.

## Models and Results
I split the data into 80% training and 20% testing (1,409 customers, 374 of them churners).
I compared a baseline that always predicts "stays" with three models.

   | Model | Accuracy | Churn precision | Churn recall | Churn F1 |
   |---|---|---|---|---|
   | Baseline (always "stays") | 0.73 | 0.00 | 0.00 | 0.00 |
   | Logistic regression | 0.81 | 0.66 | 0.57 | 0.61 |
   | Logistic regression (balanced) | 0.74 | 0.51 | 0.79 | 0.62 |
   | Random forest | 0.79 | 0.64 | 0.51 | 0.57 |
## Discussion
Accuracy is misleading here. The baseline gets 0.73 accuracy without finding any churner.
The balanced logistic regression catches more churners (recall 0.79 against 0.57) but gives more false alarms (precision 0.51 against 0.66).
Which one is better depends on the cost of a retention offer compared with the cost of losing a customer.
The random forest did not improve on logistic regression. I used default settings and did not tune it.

The balanced model missed 80 churners and caught 294. The missed churners had a longer average tenure (29 months) than the caught ones (13 months).
Among the missed churners, 46% had one- or two-year contracts. Among all churners this was only 11%.
This suggests the model mainly relies on short tenure and month-to-month contracts, and does not catch customers who look loyal but leave anyway.
With only 80 cases, this is a hint and not a proof.

## Limitations
I used one train/test split, so small differences between models may not be meaningful.
I did not tune the models.
The data has no information about why customers left.

## How to run
1. Download the dataset from Kaggle, rename it to Telco-Customer-Churn.csv (the original name is WA_Fn-UseC_-Telco-Customer-Churn.csv) and put it in the project folder.
2. Install the requirements: `pip install -r requirements.txt`
3. Run `python eda.py` for the data exploration and `python model.py` for the models.

## Notes
I used an AI assistant to explain concepts and help with errors. I wrote and ran the code myself.