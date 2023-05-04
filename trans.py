import os
import pickle

import pandas as pd

os.chdir('./learning_curves/safe_4')
for root, dirs, file in os.walk('.'):
    for path in file:
        if 'pkl' in path:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                text = pd.DataFrame(data=data)
                text.to_csv(path[:-4] + '.csv')
    break
