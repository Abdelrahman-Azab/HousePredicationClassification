from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

def splitData(X):
    list = []
    for i in X['MiscFeature2']:
        cur = i.split(',')
        arr = []
        for j in cur:
            cur2 = j.split(':') # cur2 = {'f1': 'Norm'
            #cur2 = [{'f1',Norm']
            cur2[0] = cur2[0][1:]
            cur2[1] = cur2[1][2:]
            while cur2[1][-1] == '}' or cur2[1][-1] == '\'':
                str = cur2[1]
                cur2[1] = str[:-1]
            arr.append(cur2[1])
            #cnt += 1
            #if cnt == 3:
                #cnt = 0
        list.append(arr)

    tmp = pd.DataFrame(list, columns=['f1', 'f2', 'f3'])
    X['f1'] = tmp['f1']
    X['f2'] = tmp['f2']
    X['f3'] = tmp['f3']
    X = X.drop('MiscFeature2', axis=1)
    return X
def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X