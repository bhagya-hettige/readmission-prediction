import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import itertools
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

dir = '/home/bhet0001/research/MIMIC/Data/exp/'
# dir = '/home/bhet0001/research/readmission/Med2Vec/proj/Med2Vec Model/'
# dir = '/home/bhet0001/research/readmission/project/'

df = pd.read_csv(dir + 'exp_1.csv', header=0)
# df = pd.read_csv(dir + 'exp_2_claims_codevectors.csv', header=0)
del df['Unnamed: 0']

classLabels = df[['READMITTED']]
# df_scaled = df[df.columns[df.columns.str.contains('visit_')]]
df_scaled = df
del df_scaled['READMITTED']
df_scaled[['READMITTED']] = classLabels
del df_scaled['visit']

# 'DIAG', 'PROCD',
df_scaled[['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION',
           'MARITAL_STATUS', 'ETHNICITY', 'L', 'E', 'AGE', 'DIAG', 'PROCD']] =\
    Normalizer().fit_transform(df_scaled[['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE',
                                          'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'L',
                                          'E', 'AGE', 'DIAG', 'PROCD']])
df_scaled[['GENDER']] = df_scaled[['GENDER']] - 1
df_scaled[df_scaled.columns[df_scaled.columns.str.contains('vec_')]] =\
    Normalizer().fit_transform(df_scaled[df_scaled.columns[df_scaled.columns.str.contains('vec_')]])

array = df_scaled.values

X = array[:, 0:14]
y = array[:, 14]

seed = 7
# Apply SMOTE's
kind = 'regular'
sm = SMOTE(kind='regular')
X_res, y_res = sm.fit_sample(X, y)

print("Resampled Dataset has shape: ", X_res.shape)
print("Number of Fraud Cases (Real && Synthetic): ", np.sum(y_res))

X_train, X_test, y_train, y_test= train_test_split(X, y)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

seed = 7
X_train_res, X_test_res, y_train_res, y_test_res= train_test_split(X_res, y_res)

print("")
print("Number transactions train dataset: ", len(X_train_res))
print("Number transactions test dataset: ", len(X_test_res))
print("Total number of transactions: ", len(X_train_res)+len(X_test_res))

seed = 7
est = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=1,
                                random_state=0, verbose = 1)
est.fit(X_train_res, y_train_res)

y_pred = est.predict(X_test_res)

confusion_matrix(y_test_res, y_pred)
accuracy_score(y_test_res, y_pred), 1-accuracy_score(y_test_res, y_pred)
precision_score(y_test_res, y_pred)
recall_score(y_test_res, y_pred)
print roc_auc_score(y_test_res, y_pred)


# Other classifiers
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
models.append(('SVM', SVC()))
# models.append(('GBT', GradientBoostingClassifier()))

# models.append(('MLP', MLPClassifier(alpha=1)))
# models.append(('GPC', GaussianProcessClassifier(1.0 * RBF(1.0))))
# models.append(('ABC', AdaBoostClassifier()))
# models.append(('QDA', QuadraticDiscriminantAnalysis()))

# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train_res, y_train_res, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

