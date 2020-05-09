from sklearn.feature_extraction.text import TfidfVectorizer
import os
import glob
import re
import json
import pandas as pd
import numpy as np
import copy
import pickle 
import scipy

#This function returns the numeric part of the file name and converts to an integer
def sortKeyFunc(s):
    return int(os.path.basename(s)[9:-4])


class NewTfidfVectorizer(TfidfVectorizer):

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        y : None
            This parameter is not needed to compute tfidf.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        self._check_params()
        self._warn_for_unused_params()

        #check if input is a list of lists (documents are multiple text sentences instead of single text sentences)
        if any(isinstance(el, list) for el in raw_documents):
            self._tfidf.n_sent_per_doc = []
            for i in range(len(raw_documents)):
                self._tfidf.n_sent_per_doc.append(len(raw_documents[i]))
            
            self._tfidf.n_doc = len(raw_documents)
            raw_documents2 = [item for sublist in raw_documents for item in sublist]
            assert np.cumsum(self._tfidf.n_sent_per_doc)[-1]==len(raw_documents2), "error"
        else:
            self._tfidf.n_sent_per_doc =  None

        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        y : None
            This parameter is ignored.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        self._check_params()

        #check if input is a list of lists (documents are multiple text sentences instead of single text sentences)
        if any(isinstance(el, list) for el in raw_documents):
            self._tfidf.n_sent_per_doc = []
            n_doc = 0
            for raw_document in raw_documents: #i in range(len(raw_documents)):
                self._tfidf.n_sent_per_doc.append(len(raw_document))
                n_doc += 1
            self._tfidf.n_doc = n_doc
            #raw_documents = [item for sublist in raw_documents for item in sublist]
            #assert np.cumsum(self._tfidf.n_sent_per_doc)[-1]==len(raw_documents), "error"
        else:
            self._tfidf.n_sent_per_doc =  1

        X = super().fit_transform(raw_documents)
        #self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return X#self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy="deprecated"):
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        copy : bool, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

            .. deprecated:: 0.22
               The `copy` parameter is unused and was deprecated in version
               0.22 and will be removed in 0.24. This parameter will be
               ignored.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        #check_is_fitted(self, msg='The TF-IDF vectorizer is not fitted')

        # FIXME Remove copy parameter support in 0.24
        if copy != "deprecated":
            msg = ("'copy' param is unused and has been deprecated since "
                   "version 0.22. Backward compatibility for 'copy' will "
                   "be removed in 0.24.")
            warnings.warn(msg, FutureWarning)
        
        #check if input is a list of lists (documents are multiple text sentences instead of single text sentences)
        if any(isinstance(el, list) for el in raw_documents):
            self._tfidf.n_sent_per_doc = []
            for i in range(len(raw_documents)):
                self._tfidf.n_sent_per_doc.append(len(raw_documents[i]))
            
            self._tfidf.n_doc = len(raw_documents)
            raw_documents = [item for sublist in raw_documents for item in sublist]
            assert np.cumsum(self._tfidf.n_sent_per_doc)[-1]==len(raw_documents), "error"
        else:
            self._tfidf.n_sent_per_doc =  None

        X = super().transform(raw_documents)
        return X#self._tfidf.transform(X, copy=False)

tfidf = pickle.load(open(r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\vectorizer.pk', "rb"))

#corpus = [["o mar é azul e eu vejo o horizonte."], ["o primeiro ministro está a falar.", "o primeiro ministro não gosta de gatos, mas de cães sim."],["os cães e os gatos não são para serem abandonados."], ["estou com dor de cabeça."], ["os cães são fofinhos, os gatos não."], ["os cães comem ração."], ["o primeiro ministro está com dor de cabeça."]]


#train_corpus = corpus[0:4]
#dev_corpus = corpus[4:6]
#test_corpus = corpus[6]

#max_features = 10
#max_features_ = [5,2]

#tfidf = NewTfidfVectorizer(max_features=max_features)

#Fit transform with train
#train_feature_matrix = tfidf.fit_transform(train_corpus)
train_feature_matrix = scipy.sparse.load_npz(r"C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\train_tfidf.npz")#np.load(r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\train_tfidf.npz')

n_features = tfidf._tfidf._idf_diag.shape[0]
assert n_features == train_feature_matrix.shape[1], "The number of features in the training matrix is not equal to the number of features in the vectorizer"

max_features_=[768,200]

#Dev and test
#Change number of features
print(tfidf.vocabulary_)
original_vectorizer = copy.deepcopy(tfidf) #save the original tfidf vectorizer

for i in range(len(max_features_)):
    train_feature_matrix2, removed_terms = tfidf._limit_features(train_feature_matrix, vocabulary=tfidf.vocabulary_, limit=max_features_[i])
    #tfidf now is the new tfidf with max_features*
    print(tfidf.vocabulary_)

    dev_feature_matrix = tfidf.transform(dev_corpus)
    test_feature_matrix = tfidf.transform([test_corpus])

    #update
    del tfidf
    tfidf = copy.deepcopy(original_vectorizer)


print("hello")