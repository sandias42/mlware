""" This file implements completely naive featurization of the xml files."""
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
import os
from scipy import io
from optparse import OptionParser

train_paths = os.listdir("../data/train/")
test_paths = os.listdir("../data/test/")

train_ids = []
train_class = []
test_ids = []

"""# parse command line argument
op = OptionParser()
op.add_option("--vectorizer",
              action="store_true",
              help="Use sklearn's HashingVectorizer to extract features")
op.add_option("--str_process",default="""


def generate_xml_paths(train_paths, test_paths, xml_processor=lambda x: x, i=0):
    """ 
    Processes the provided paths, extracting id and class information and 
    applying whatever function on the xml is desired.
    xml_processor should takes in xml_string and should return something
    """
    paths = train_paths + test_paths
    print "The length of the test data is {0}, training data {1}".format(
        len(test_paths), len(train_paths)
    )
    while i < len(paths):
        abs_path = ''
        # Split the file name into a list of [id, class_name, xml]
        id_class_xml = paths[i].split('.')
        assert id_class_xml[2] == 'xml'

        # If the file is part of the test set, append the id to test_ids
        if i >= len(train_paths):
            test_ids.append(id_class_xml[0])
            assert id_class_xml[1] == 'X'
            abs_path = os.path.join(
                os.path.abspath("../data/test/"), paths[i])

        # Otherwise file is in training set. Append id and class
        else:
            train_ids.append(id_class_xml[0])
            train_class.append(id_class_xml[1])
            abs_path = os.path.join(
                os.path.abspath("../data/train/"), paths[i])

        # Open the file, process, and yield string
        with open(abs_path, 'r') as xml_file:
            xml_content = xml_processor(xml_file.read())
            assert type(xml_content) == str
            yield xml_content
            print "sent file {0}, named \n {1} to processing".format(i, paths[i])
            i += 1


# First try producing features with Hashing Vectorizer,
# Which returns a scipy_sparse matrix with shape
# (n_samples, 2 ** 20 features). Has some downsides and
# may not be useable in training

# first use simple word tokens (whitespace sperated?)
word_hasher = HashingVectorizer()

hashed_sparse_mat = word_hasher.transform(
    generate_xml_paths(train_paths, test_paths)
)

print hashed_sparse_mat

print type(hashed_sparse_mat)

# Save the matrix as follows
io.mmwrite("../data/features/naive_word_hashed_full_features.mtx",
           hashed_sparse_mat)

np.save("../data/features/test_ids.npy", np.array(test_ids))
np.save("../data/features/train_ids.npy", np.array(train_ids))
np.save("../data/features/train_classes.npy", np.array(train_class))
