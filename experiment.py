__author__ = 'Stephen Zakrewsky'

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_curve

with open('ds.json') as dsh:
    ds = json.load(dsh)

y = np.empty((len(ds),))
data = np.empty((len(ds), 5158))
for i, d in enumerate(ds):
    y[i] = d['views'] + d['num_favorers']
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
    data[i][38:] = d['Khosla14-texture']

y = (y - y.min())/y.max()
median = np.median(y)
print "Median ", median
labels = y >= median
print 'n=', len(labels), '+=', np.count_nonzero(labels)

cv = StratifiedKFold(labels, n_folds=10)
lr = LogisticRegression()
# train_data, test_data, train_labels, test_labels = train_test_split(data, labels)
# lr.fit(train_data, train_labels)
# fpr, tpr, _ = roc_curve(test_labels, lr.predict_proba(test_data)[:,1])


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
precision_50 = 0
recall_50 = 0
precision_median = 0
recall_median = 0
f1_50 = 0
f1_median = 0
mse = 0
mae = 0
for i, (train, test) in enumerate(cv):
    print "Running round", i
    probas_ = lr.fit(data[train], labels[train]).predict_proba(data[test])
    precision_50 += precision_score(labels[test], probas_[:, 1] >= 0.5)
    recall_50 += recall_score(labels[test], probas_[:, 1] >= 0.5)
    precision_median += precision_score(labels[test], probas_[:, 1] >= median)
    recall_median += recall_score(labels[test], probas_[:, 1] >= median)
    f1_50 += f1_score(labels[test], probas_[:, 1] >= 0.5)
    f1_median += f1_score(labels[test], probas_[:, 1] >= median)
    mse += mean_squared_error(y[test], probas_[:, 1])
    mae += mean_absolute_error(y[test], probas_[:, 1])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(labels[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

print "Precision 0.5", precision_50/10.0, "Precision median", precision_median/10.0
print "Recall 0.5", recall_50/10.0, "Recall median", recall_median/10.0
print "F1 0.5", f1_50/10.0, "F1 median", f1_median/10.0
print "MSE", mse/10.0, "MAE", mae/10.0

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()