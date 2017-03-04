from gensim import corpora
import gensim
import re
import os
import numpy as np
from six import iteritems

#converts xml string to list of words, removing \r\n and ", splitting on whitespace, < and >
def str_to_list(r):
    return filter(None, re.split("[, <>]+", re.sub(r'"|\r\n', "", r)))

train_paths = os.listdir("../data/train/")
test_paths = os.listdir("../data/test/")

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

dictionary = corpora.Dictionary(sentences)
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq < 4]
dictionary.filter_tokens(once_ids)  # remove words that appear only once
dictionary.compactify()

corpus = [dictionary.doc2bow(s, allow_update=False) for s in sentences]
#np.save('../data/bow.npy', gensim.matutils.corpus2csc(corpus).todense().T)

tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
#np.save('../data/tfidf.npy', gensim.matutils.corpus2csc(corpus_tfidf).todense().T)

lsi = gensim.models.LsiModel(corpus_tfidf, num_topics=500)
corpus_lsi = lsi[corpus_tfidf]

np.save('../data/lsi_500.npy', gensim.matutils.corpus2csc(corpus_lsi).todense().T)
#print(np.load('lsi.npy').shape)
