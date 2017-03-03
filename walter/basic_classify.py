from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.io import mmread
import numpy as np

malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]

def classes_to_Y(classes):
    output = []
    for cls in classes:
        output.append(malware_classes.index(cls))
    return np.array(output)

# load training classes
classes = np.load("../data/features/train_classes.npy")

# load sparse matrix of training data
sparse_mat_train_test = mmread("../data/features/naive_word_hashed_full_features.mtx")

print('data is loaded')

# convert csr to a numpy array
sparse = sparse_mat_train_test.toarray()

# pull out training examples
X = sparse[:classes.shape[0]]
X_CV = X[:1000]
Y = classes_to_Y(classes)
Y_CV = Y[:1000]

RF = RandomForestClassifier()
RF.fit(X, Y)
print 'done fitting'
preds = RF.predict(X_CV)

mistakes = 0
for i in range(len(preds)):
    if preds[i] != Y_CV[i]:
        mistakes += 1
print mistakes
# print cross_val_score(RF, X, Y, cv=2, scoring="mean_squared_error")
