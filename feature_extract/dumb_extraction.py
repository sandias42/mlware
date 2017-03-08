
""" This file implements completely naive featurization of the xml files."""
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
import os
from scipy import io
from optparse import OptionParser
import sys

train_paths = os.listdir("../data/train/")
test_paths = os.listdir("../data/test/")

train_ids = []
train_class = []
test_ids = []

# parse command line arguments
op_parse = OptionParser()
op_parse.add_option("--vectorizer", action="store",
                    type=str, default="hashing",
                    help="Specify the vectorizer to extract features. Sklearn's HashingVectorizer is default")
op_parse.add_option("--str_processing", default="naive", action="store",
                    help="Specify the method to parse xml. Default is no processing")
op_parse.add_option("--extract_id", action="store_true",
                    help="Extract and save the id from test and training data")
op_parse.add_option("--extract_class", action="store_true",
                    help="Extract and save the class label of each class in the training set")

(op, args) = op_parse.parse_args()
if len(args) > 0:
    op.error("This script takes no arguments")
    sys.exit(1)

    op_parse.print_help()


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
            print 'heres a thing'
            if op.extract_id:
                test_ids.append(id_class_xml[0])
            assert id_class_xml[1] == 'X'
            abs_path = os.path.join(
                os.path.abspath("../data/test/"), paths[i])

        # Otherwise file is in training set. Append id and class
        else:
            if op.extract_id:
                train_ids.append(id_class_xml[0])
            if op.extract_class:
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
if op.vectorizer == "hashing":
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

elif op.vectorizer == "hash_4gram_tfidf":
    # pipe vectorizer with ngrams and tfidf
    pipe = make_pipeline(
        HashingVectorizer(ngram_range=(1, 4)),
        TfidfTransformer()
    )
    hashed_sparse_mat = pipe.fit_transform(
        generate_xml_paths(train_paths, test_paths)
    )

    print hashed_sparse_mat
    print type(hashed_sparse_mat)
    # Save the matrix as follows
    io.mmwrite("../data/features/tfifd_4gram_hashed_full_features.mtx",
               hashed_sparse_mat)

elif op.vectorizer == "counts10000":
    word_vectorizer = CountVectorizer(max_features=10000, vocabulary=None)
    path_gen = generate_xml_paths(train_paths, test_paths)
    count_vec_corpus = word_vectorizer.fit_transform(path_gen).toarray()
    np.save("../data/features/count_vector_full_10k_features.npy",
            np.array(count_vec_corpus)
            )

elif op.vectorizer == "counts_tfidf10000":
    pipe = make_pipeline(CountVectorizer(
        max_features=10000), TfidfTransformer())
    normalized_corpus = pipe.fit_transform(
        generate_xml_paths(train_paths, test_paths)
    ).toarray()
    np.save("../data/features/count_vector_full_10k_features_tfidf.npy",
            np.array(normalized_corpus)
            )

elif op.vectorizer == "none":
    [0 for __ in generate_xml_paths(train_paths, test_paths)]

if op.extract_id:
    np.save("../data/features/test_ids.npy", np.array(test_ids))
    np.save("../data/features/train_ids.npy", np.array(train_ids))

if op.extract_class:
    np.save("../data/features/train_classes.npy", np.array(train_class))
