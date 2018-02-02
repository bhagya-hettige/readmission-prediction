import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

dir = '/home/bhet0001/research/readmission/Med2Vec/proj/Med2Vec Model Demo/'

df = pd.read_csv(dir + 'claims_codevectors.csv', header=0)
del df['Unnamed: 0']
df = df[df.columns[~df.columns.str.contains('ICD9_')]]
array = df.values

raw_data = array[:, 0:218]
raw_target = array[:, 218]

train, test, train_t, test_t = train_test_split(raw_data, raw_target, test_size=0.2, random_state=2)

common_args = {'max_depth': 3, 'n_estimators': 500, 'subsample': 0.9, 'random_state': 2}

models = [('max', GradientBoostingClassifier(learning_rate=1, **common_args)),
          ('high', GradientBoostingClassifier(learning_rate=0.5, **common_args)),
          ('low', GradientBoostingClassifier(learning_rate=0.05, **common_args)),
         ]
stage_preds = {}
final_preds = {}

for mname, m in models:
    m.fit(train, train_t)
    stage_preds[mname] = {'train': list(m.staged_predict_proba(train)),  'test': list(m.staged_predict_proba(test))}
    final_preds[mname] = {'train': m.predict_proba(train),  'test': m.predict_proba(test)}

def frame(i=0, log=False):
    for mname, _ in models:
        plt.hist(stage_preds[mname]['train'][i][:,1], bins=np.arange(0,1.01,0.01), label=mname, log=log)
    plt.xlim(0,1)
    plt.ylim(0,8000)
    if log:
        plt.ylim(0.8,10000)
        plt.yscale('symlog')
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
    plt.legend(loc='upper center')
    return

plt.figure(figsize=(16,10))
for pos, fnum in enumerate((1, 5, 50, 100), 0):
    plt.subplot2grid((3,2), (pos/2, pos%2))
    frame(fnum-1, True)
    plt.title("Predictions for each model at tree #%d (y axis on log scale)" % fnum)

plt.subplot2grid((3,2), (2,0), colspan=2)
plt.title("Final predictions for each model (y axis on linear scale)")
frame(-1, False)
plt.ylim(0,7000)
plt.show()

plt.figure(figsize=(12,6))
for marker, (mname, preds) in zip(["-", "--", ":"], stage_preds.iteritems()):
    for c, (tt_set, target) in zip(['#ff4444', '#4444ff'], [('train', train_t), ('test', test_t)]):
        aucs = map(lambda x: roc_auc_score(target, x[:,1]), preds[tt_set])
        label = "%s: %s" % (mname, tt_set) + (" (best: %.3f @ tree %d)" % (max(aucs), np.array(aucs).argmax()+1) if tt_set == 'test' else "")
        plt.plot(aucs, marker, c=c, label=label)
plt.ylim(0.93, 1)
plt.title("Area under ROC curve (AUC) for each tree in each GBM")
plt.xlabel("Tree #")
plt.ylabel("AUC")
plt.legend(loc="lower center")
plt.show()
