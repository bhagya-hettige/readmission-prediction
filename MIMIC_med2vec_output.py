import numpy as np
import pandas as pd
import math
import theano
import theano.tensor as T

dir = '/home/bhet0001/research/MIMIC/Data/exp/'

loaded = np.load(dir + 'exp_2_med2vec_output.9.npz')

df = pd.read_csv(dir + 'exp_2.csv')

del df['Unnamed: 0']

for i in range(0, len(df)):
    record = df.loc[i]
    x = np.zeros(10661)
    codes = eval(record[13])
    for code in codes:
        if not isinstance(code, str) and not math.isnan(float(code)):
            code = int(code)
            if code < 10661:
                x[code] = 1
    emb = (np.dot(x, loaded['W_emb']) + loaded['b_emb']).clip(min=0)
    # if options['demoSize'] > 0: emb = T.concatenate((emb, d), axis=1)
    visit = (np.dot(emb, loaded['W_hidden']) + loaded['b_hidden']).clip(min=0)
    # results = T.nnet.softmax(T.dot(visit, tparams['W_output']) + tparams['b_output'])

    # visits.append(visit)
    # print(np.array(visit.tolist()))
    for j in range(0, 200):
        df.at[i, 'vec_' + str(j)] = visit[j]

df.to_csv(dir + 'exp_2_claims_codevectors.csv')
print (df)
