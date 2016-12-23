__author__ = 'Stephen Zakrewsky'

import ijson.backends.yajl2_cffi as ijson
import numpy as np

n_samples = 3305
n_features = 42022
tsz = np.empty((n_samples,))
y = np.empty((n_samples,))
data = np.empty((n_samples, n_features))

print 'Loading dataset...'
with open('../workspace/ds_deep.json') as inh:
    ds = ijson.items(inh, 'item')
    for i, d in enumerate(ds):
        print 'Processing', i + 1, 'out of', n_samples
        y[i] = d['views'] + d['num_favorers']
        tsz[i] = d['original_creation_tsz']
        data[i][0] = d['Ke06-qa']
        data[i][1] = d['Ke06-qh']
        data[i][2] = d['Ke06-qf']
        data[i][3] = d['Ke06-tong']
        data[i][4] = d['Ke06-qct']
        data[i][5] = d['Ke06-qb']
        data[i][6] = d['-mser_count']
        data[i][7:32] = d['Mai11-thirds_map']
        data[i][32] = d['Wang15-f1']
        data[i][33] = d['Wang15-f14']
        data[i][34] = d['Wang15-f18']
        data[i][35] = d['Wang15-f21']
        data[i][36] = d['Wang15-f22']
        data[i][37] = d['Wang15-f26']
        data[i][38:5158] = d['Khosla14-texture']
        data[i][5158:42022] = d['deep']

print 'Saving dataset...'
np.savez('../workspace/ds', tsz=tsz, y=y, data=data)