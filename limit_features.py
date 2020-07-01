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
import argparse
import pickle
import pdb
import create_tfidf_per_dt as run

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'NewTfidfVectorizer':
            from create_tfidf_per_dt import NewTfidfVectorizer
            return NewTfidfVectorizer
        return super().find_class(module, name)

#This function returns the numeric part of the file name and converts to an integer
def sortKeyFunc(s):
    return int(os.path.basename(s)[9:-4])

def main(args):

    output_dir = args.output_dir

    #Load vectorizer
    if '/' in args.vectorizer_path:
        vectorizer_path = args.vectorizer_path+'/'
        output_dir = output_dir+'/' 
    else:
        vectorizer_path = args.vectorizer_path+'\\'
        output_dir = output_dir+'\\'
    
    subfolders_path = [output_dir+name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,name)) and '.' in name]

    tfidf = CustomUnpickler(open(vectorizer_path+ 'vectorizer.pk', 'rb')).load()
    #tfidf = pickle.load(open(vectorizer_path, "rb"))

    #Load train feature matrix
    train_feature_matrix = scipy.sparse.load_npz(vectorizer_path+'train_tfidf.npz')
    dev_feature_matrix = scipy.sparse.load_npz(vectorizer_path+'dev_tfidf.npz')
    test_feature_matrix = scipy.sparse.load_npz(vectorizer_path+'test_tfidf.npz')

    #Get dev and test corpus from vectorizer
    dev_corpus = tfidf.inverse_transform(dev_feature_matrix)
    test_corpus = tfidf.inverse_transform(test_feature_matrix)
    
    dev_corpus = [' '.join(dev_corpus[i]) for i in range(len(dev_corpus))]
    test_corpus = [' '.join(test_corpus[i]) for i in range(len(test_corpus))]

    n_features = tfidf._tfidf._idf_diag.shape[0]
    new_n_features = args.n_features

    original_vectorizer = copy.deepcopy(tfidf) #save the original tfidf vectorizer

    for i in range(len(new_n_features)):
        
        #pdb.set_trace()
        assert new_n_features[i]<n_features, "It is not possible to limit the number of features to a number greater than max_features= " + str(n_features)+" used during training."

        #Dev and test
        #Change number of features
        #print(tfidf.vocabulary_)
        train_feature_matrix2, removed_terms = tfidf._limit_features(train_feature_matrix, vocabulary=tfidf.vocabulary_, limit=new_n_features[i])
        #tfidf now is the new tfidf with new_n_features
        #print(tfidf.vocabulary_)

        dev_feature_matrix2 = tfidf.transform(dev_corpus)
        test_feature_matrix2 = tfidf.transform(test_corpus)

        #Store train, dev and test matrices with reduced number of features (=new_n_features)
        if not os.path.exists(vectorizer_path+str(new_n_features[i])):
            os.makedirs(vectorizer_path+str(new_n_features[i]))
        
        if '/' in output_dir:
            path = output_dir+ str(new_n_features[i])+'/'
        else:
            path = output_dir+ str(new_n_features[i])+'\\'
            

        scipy.sparse.save_npz(os.path.join(path, "train_tfidf.npz"), train_feature_matrix2)
        scipy.sparse.save_npz(os.path.join(path, "dev_tfidf.npz"), dev_feature_matrix2)
        scipy.sparse.save_npz(os.path.join(path, "test_tfidf.npz"), test_feature_matrix2)

        #Store vectorizer with reduced number of features (=new_n_features)
        with open(os.path.join(path, 'vectorizer.pk'), 'wb') as infile:
            pickle.dump(tfidf, infile)
        
        #List all windows
        os.chdir(vectorizer_path)
        result = glob.glob('*/')
        #subfolders_path = [folder for folder in result if folder!='.' and folder!='..'] 
        #subfolders_path = [folder for folder in result if folder!='.' and folder!='..'] 
        #subfolders_path = [x[0] for x in os.walk(output_dir) if x[0]!= output_dir and x[0]!=path[:-1]]

        for window in subfolders_path:
            dw = int(window[-1])
            dt = int(window[-3])

            #Run for all windows again
            print("Creating dataset for discretization unit "+str(dt)+" and window size "+str(dw)+"...")

            if '/' in path:
                #output_dir_dt = path + str(dt)+'/'
                output_dir_dtdw = path + str(dt)+'.'+str(dw)+'/'
            else:
                #output_dir_dt = path + str(dt)+'\\'
                output_dir_dtdw = path + str(dt)+'.'+str(dw)+'\\'

            #if not os.path.exists(output_dir_dt):
            #    os.makedirs(output_dir_dt)

            if not os.path.exists(output_dir_dtdw):
                os.makedirs(output_dir_dtdw)
            
            args.discretization_unit = dt
            args.window_size = dw
            args.create = False
            args.output_dir = path#output_dir_dt
            args.ids_path = None
            args.max_features = new_n_features[i]
            args.csv_path = None
        
            run.main(args)

        #update
        del tfidf
        del train_feature_matrix2
        del dev_feature_matrix2
        del test_feature_matrix2


        tfidf = copy.deepcopy(original_vectorizer)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Create tf idfs of a tweet dataset to be used as embeddings.')
    
    parser.add_argument('--vectorizer_path', default=r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\dt\1\\', help="Full path where the train tfidf vectorizer is located")
    parser.add_argument('--n_features', default=100000, help="Number of features to consider in the TfIdfVectorizer. Must be less than the number used during training.") #It can be a number or a vector
    parser.add_argument('--output_dir', default=r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\dt\1\\', help="Output dir to store the tweet times and tf idfs of train dev and test.")
    args = parser.parse_args()
    
    print(args) 
