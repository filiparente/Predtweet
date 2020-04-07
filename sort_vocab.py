import pickle
import pandas as pd
import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer

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
            for i in range(len(raw_documents)):
                self._tfidf.n_sent_per_doc.append(len(raw_documents[i]))
            
            self._tfidf.n_doc = len(raw_documents)
            raw_documents = [item for sublist in raw_documents for item in sublist]
            assert np.cumsum(self._tfidf.n_sent_per_doc)[-1]==len(raw_documents), "error"
        else:
            self._tfidf.n_sent_per_doc =  1

        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

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
        check_is_fitted(self, msg='The TF-IDF vectorizer is not fitted')

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
        return self._tfidf.transform(X, copy=False)


#corpus = [['This is the first document.', 'This is also the first document.'], ['And this is the second one.'], ['Is this the first document? No, third.']]
#vectorizer = NewTfidfVectorizer()
#X = vectorizer.fit(corpus)



vectorizer = pickle.load(open(r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\dt\1\vectorizer.pk', "rb"))

print("done!")

#Sort vectorizer by idf's (>=1) of train: a lower idf indicates a word that occurs a lot in the train documents, an idf of 1 means that the corresponding word appears in all train documents!
indices = np.argsort(vectorizer.idf_)

list_feature_names = vectorizer.get_feature_names()

features = []
idfs = []
for i in range(len(indices)):
    #Extract feature name
    features.append(list_feature_names[indices[i]])

    #Extract 
    idfs.append(vectorizer.idf_[indices[i]])

d = {'IDF': idfs, 'Word': features}
df = pd.DataFrame(data=d)
df.to_csv(r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\dt\1\IDF_train.csv')

            
