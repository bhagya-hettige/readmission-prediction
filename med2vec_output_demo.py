import numpy as np
import pandas as pd
import math
import theano
import theano.tensor as T

dir = '/home/bhet0001/research/readmission/Med2Vec/proj/Med2Vec Model Demo/'

loaded = np.load(dir + 'med2vec_output_demo.9.npz')

df = pd.read_csv(dir + 'claims.csv')

del df['Unnamed: 0']
del df['PATIENT_ID']
classLabel = df['READMITTED']
del df['READMITTED']

for i in range(0, len(df)):
    record = df.loc[i]
    x = np.zeros(5521)
    codes = record[3:19]
    demo = np.concatenate((record[0:3], record[19:34]), axis=0)
    for code in codes:
        if not isinstance(code, str) and not math.isnan(float(code)):
            code = int(code)
            if code < 5521:
                x[code - 1] = 1
    emb = (np.dot(x, loaded['W_emb']) + loaded['b_emb']).clip(min=0)
    emb = np.concatenate((emb, demo), axis=0)
    visit = (np.dot(emb, loaded['W_hidden']) + loaded['b_hidden']).clip(min=0)
    # results = T.nnet.softmax(T.dot(visit, tparams['W_output']) + tparams['b_output'])

    # visits.append(visit)
    # print(np.array(visit.tolist()))
    for j in range(0, 200):
        df.at[i, 'visit_' + str(j)] = visit[j]

df['READMITTED'] = classLabel
df.to_csv(dir + 'claims_codevectors.csv')
print (df)
