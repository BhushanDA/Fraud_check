import pandas as pd
# Reading the Fraud Data #################
fraud = pd.read_csv(r"D:\Python\Fraud_check.csv")
fraud.head()
fraud.columns

fraud["TI"] = pd.cut(fraud["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
FraudCheck = fraud.drop(columns=["Taxable.Income"])

FCheck = pd.get_dummies(FraudCheck.drop(columns = ["TI"]))

FraudC_final = pd.concat([FCheck, FraudCheck["TI"]], axis = 1)
colnames = list(FraudC_final.columns)
predictors = colnames[:9]
target = colnames[9]

X = FraudC_final[predictors]
Y = FraudC_final[target]

from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions
model.fit(X,Y)
model.pred=model.predict(X)

from sklearn.metrics import confusion_matrix
confusion_matrix(FraudC_final['TI'],model.pred) # Confusion matrix

print("Accuracy",(476+124)/(476+124+0+0)*100)#100%

