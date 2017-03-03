""" This file implements completely naive featurization of the xml files."""
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
import os
import unittest
from scipy import io

train_paths = os.listdir("../data/train/")
test_paths = os.listdir("../data/train/")

train_ids = []
train_class = []
test_ids = []


def generate_xml_paths(train_paths, test_paths, xml_processor=lambda x: x, i=0):
    """ 
    Processes the provided paths, extracting id and class information and applying
    whatever function on the xml is desired.
    """
    paths = train_paths + test_paths
    while i <= len(paths):

        # Split the file name into a list of [id, class_name, xml]
        id_class_xml = paths[i].split('.')
        unittest.assertEqual(id_class_xml[2], 'xml')

        # If the file is part of the test set, append the id to test_ids
        if i > len(train_paths):
            test_ids.append(id_class_xml[0])
            unittest.assertEqual(id_class_xml[1], 'X')

        # Otherwise file is in training set. Append id and class
        else:
            train_ids.append(id_class_xml[0])
            train_class.append(id_class_xml[1])

        # Open the file, process, and yield string
        with open(paths[i], 'r') as xml_file:
            xml_content = xml_processor(xml_file.read())
            unittest.asserIsInstance(xml_content, str)
            yield xml_content
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

np.array(test_ids).save("../data/features/test_ids.np")
np.array(train_ids).save("../data/features/train_ids.np")
np.array(train_class).save("../data/features/train_classes.np")
