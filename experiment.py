__author__ = 'Stephen Zakrewsky'

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import auc, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import scale

print 'Loading dataset...'
ds = np.load('../workspace/ds.npz')

tsz = ds['tsz']
y = ds['y']
data = ds['data']

y = y * np.power(tsz/np.min(tsz), 50)
y = (y - y.min())/y.max()
median = np.median(y)
mp = median + 0.0027
mn = median - 0.001
print "Median ", median, "+", mp, "-", mn
print "count", len(y), "+", np.count_nonzero(y >= median), "-", np.count_nonzero(y < median)
print "count +-", np.count_nonzero(y > mp) + np.count_nonzero(y < mn), "+", np.count_nonzero(y > mp), "-", np.count_nonzero(y < mn)
within_threshold = np.logical_or(y > mp, y < mn)
data = data[within_threshold]
y = y[within_threshold]
labels = y >= median
print "len(data)", len(data), "len(y)", len(y)

cv = StratifiedKFold(labels, n_folds=10, shuffle=True)
r = LogisticRegression(C=0.001)
#r = KNeighborsRegressor(n_neighbors=2)
#r = DecisionTreeRegressor()
#r = KernelRidge(kernel='rbf', alpha=0.0001)
#r = Ridge(alpha=100)

data = scale(data)

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
    #probas_ = r.fit(data[train], y[train]).predict(data[test])
    probas_ = r.fit(data[train], labels[train]).predict_proba(data[test])[:, 1]
    precision_50 += precision_score(labels[test], probas_ >= 0.5)
    recall_50 += recall_score(labels[test], probas_ >= 0.5)
    precision_median += precision_score(labels[test], probas_ >= median)
    recall_median += recall_score(labels[test], probas_ >= median)
    f1_50 += f1_score(labels[test], probas_ >= 0.5)
    f1_median += f1_score(labels[test], probas_ >= median)
    mse += mean_squared_error(y[test], probas_)
    mae += mean_absolute_error(y[test], probas_)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(labels[test], probas_)
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