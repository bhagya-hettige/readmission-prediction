import pickle
import numpy as np

dir = '/home/bhet0001/research/readmission/Med2Vec/proj/'
dir2 = '/home/bhet0001/research/MIMIC/Data/exp/'

# output.txt - medical concepts
# Unique medical codes = 18510
with open(dir2 + 'exp_3_input.txt') as file:
    arr = eval(file.read())
    arr = arr[:10000]
    with open('exp_3_input_formatted.pkl', 'w') as f:
        pickle.dump(arr, f)

# demo.txt - demographic details of patients
# with open(dir2 + 'exp_3_demo.txt') as file:
#     st = file.read()
#     mat = eval(st)
#     s = ''
#     for m in mat:
#         s += " ".join(map(str, m)) + '; '
#     s = s[0:-2]
#
#     mat = np.matrix(s)
#     with open('exp_3_demo_formatted.pkl', 'w') as f:
#         pickle.dump(mat, f)
