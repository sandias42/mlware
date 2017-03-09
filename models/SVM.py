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

# convert csr to a numpy array
sparse = np.load("/n/regal/scrb152/Students/sandias42/cs181/bow.npy")

# pull out training examples
X = sparse[:classes.shape[0],:]

X_test = sparse[classes.shape[0]:,:]
print X_test.shape

Y = classes_to_Y(classes)

model = SGDClassifier(n_jobs=-1, n_iter=100, verbose=1, loss="modified_huber")
model.fit(X,Y)

test_pred = model.predict(X_test)

print test_pred
test_ids = np.load("../data/features/test_ids.npy")
print test_ids
write_predictions(test_pred, test_ids, "../predictions/sgd_bow.csv")
