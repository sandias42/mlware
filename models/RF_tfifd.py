from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
#from sklearn.model_selection import cross_val_score
from scipy.io import mmread
import numpy as np

malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]

# a function for writing predictions in the required format
def write_predictions(predictions, ids, outfile):
    """
    assumes len(predictions) == len(ids), and that predictions[i] is the
    index of the predicted class with the malware_classes list above for
    the executable corresponding to ids[i].
    outfile will be overwritten
    """
    with open(outfile,"w+") as f:
        # write header
        f.write("Id,Prediction\n")
        for i, history_id in enumerate(ids):
            f.write("%s,%d\n" % (history_id, predictions[i]))

def classes_to_Y(classes):
    output = []
    for cls in classes:
        output.append(malware_classes.index(cls))
    return np.array(output)

# load training classes
classes = np.load("../data/features/train_classes.npy")

# load sparse matrix of training data
sparse_mat_train_test = mmread("../data/features/tfifd_4gram_hashed_full_features.mtx")

# convert csr to a numpy array
sparse = sparse_mat_train_test.toarray()

# pull out training examples
X = sparse[:classes.shape[0],:]

X_test = sparse[classes.shape[0]:,:]
print X_test.shape

Y = classes_to_Y(classes)

RF = RandomForestClassifier(n_jobs=-1, verbose=42, n_estimators=100)
RF.fit(X, Y)

test_pred = RF.predict(X_test)


print test_pred
test_ids = np.load("../data/features/test_ids.npy")
print test_ids
write_predictions(test_pred, test_ids, "../predictions/rfc_100_big_tfifd.csv")
