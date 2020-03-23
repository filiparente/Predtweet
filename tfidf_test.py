from sklearn.feature_extraction.text import TfidfVectorizer
import os
import glob
import re
import json
import pandas as pd

#This function returns the numeric part of the file name and converts to an integer
def sortKeyFunc(s):
    return int(os.path.basename(s)[9:-4])


corpus = ["o mar é azul e eu vejo o horizonte.", "o primeiro ministro está a falar.","os cães e os gatos não são para serem abandonados.", "estou com dor de cabeça.", "os cães são fofinhos, os gatos não.", "os cães comem ração.", "o primeiro ministro está com dor de cabeça."]


train_corpus = corpus[0:4]
dev_corpus = corpus[4:6]
test_corpus = corpus[6]

tfidf = TfidfVectorizer()

#Fit transform with train
train_feature_matrix = tfidf.fit_transform(train_corpus)
print(list(tfidf.get_params()))

#Dev and test
dev_feature_matrix = tfidf.transform(dev_corpus)
test_feature_matrix = tfidf.transform([test_corpus])

print("hello")