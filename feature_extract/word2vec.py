from gensim import corpora
import gensim
import re
import os
import numpy as np
from six import iteritems

#converts xml string to list of words, removing \r\n and ", splitting on whitespace, < and >
def str_to_list(r):
    return filter(None, re.split("[, <>]+", re.sub(r'"|\r\n', "", r)))

train_paths = os.listdir("../data/tr/")
test_paths = os.listdir("../data/te/")

train_ids = []
train_class = []
test_ids = []

   
paths = train_paths + test_paths
abs_paths = []

for i in range(len(paths)):
    abs_path = ''

    if i >= len(train_paths):
        abs_path = os.path.join(
            os.path.abspath("../data/test/"), paths[i])

    else:
        abs_path = os.path.join(
            os.path.abspath("../data/train/"), paths[i])
    abs_paths.append(abs_path)

class MySentences(object):
    def __iter__(self):
        for path in abs_paths:
            yield str_to_list(open(path).read())

sentences = MySentences()

model = gensim.models.word2vec.Word2Vec(sentences, size=400, window=5, min_count=5, workers=10)
print model.syn0.shape
np.save('../data/w2v_400.npy', model.syn0)
