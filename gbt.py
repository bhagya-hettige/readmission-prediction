import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


dir = '/home/bhet0001/research/readmission/project/'

train_set = pd.read_csv(dir + 'admission_claims_train.csv', header=0)
test_set = pd.read_csv(dir + 'admission_claims_test.csv', header=0)

final_train = train_set
final_test = test_set

del final_train["Unnamed: 0"]
del final_test["Unnamed: 0"]

y_train = final_train.pop('READMITTED')
y_test = final_test.pop('READMITTED')

cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic', 'max_depth': 3, 'min_child_weight': 1}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                            cv_params,
                             scoring = 'accuracy', cv = 5, n_jobs = -1)

print (optimized_GBM.fit(final_train, y_train))
print (optimized_GBM.grid_scores_)

xgdmat = xgb.DMatrix(final_train, y_train)

our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1}
# Grid Search CV optimized settings

cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 3000, nfold = 5,
                metrics = ['error'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error

print(cv_xgb.tail(5))

our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1}

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 432)
sns.set(font_scale = 1.5)
xgb.plot_importance(final_gb)

importances = final_gb.get_fscore()
importances

importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)
importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')

testdmat = xgb.DMatrix(final_test)
y_pred = final_gb.predict(testdmat) # Predict using our testdmat

y_pred[y_pred > 0.7] = 1
y_pred[y_pred <= 0.7] = 0

accuracy_score(y_test, y_pred), 1-accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
