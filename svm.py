import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


dir = '/home/bhet0001/research/readmission/project/'

train_set = pd.read_csv(dir + 'admission_claims_train.csv', header=0)
test_set = pd.read_csv(dir + 'admission_claims_test.csv', header=0)

final_train = train_set
final_test = test_set

del final_train["Unnamed: 0"]
del final_test["Unnamed: 0"]

y_train = final_train['READMITTED']
del final_train['READMITTED']

y_test = final_test['READMITTED']
del final_test['READMITTED']

clf = svm.NuSVC(nu=0.1)
clf.fit(final_train, y_train)
y_pred = clf.predict(final_test)

print accuracy_score(y_test, y_pred), 1-accuracy_score(y_test, y_pred)
print precision_score(y_test, y_pred), recall_score(y_test, y_pred), confusion_matrix(y_test, y_pred)
