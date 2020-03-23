import torch
from transformers import BertModel, BertTokenizer, BertTokenizerFast
import numpy as np
from datetime import timedelta
import pandas as pd 
from torch.nn.utils.rnn import pad_sequence
import cProfile
import progressbar
from time import sleep
from sentence_transformers import SentenceTransformer
import os
import glob
import re
import datetime
import json
from langdetect import detect
import csv
import argparse
import matplotlib.pyplot as plt
import pickle
import scipy
from scipy.io import savemat
import csv

#from pyTweetCleaner import TweetCleaner
from preprocess import twokenize
from pathlib import Path
import pycld2 as cld2
from sklearn.feature_extraction.text import TfidfVectorizer

#This function returns the numeric part of the file name and converts to an integer
def sortKeyFunc(s):
    return int(os.path.basename(s)[9:-4])

#current path
cpath = Path.cwd()

# Start profiling code
pr = cProfile.Profile()
pr.enable()
class TweetBatch():
    def __init__(self, discretization_unit, window_size):
        dataset = {}
        dataset['disc_unit'] = discretization_unit
        dataset['window_size'] = window_size
        dataset['input_ids'] = []

        self.delta = timedelta(hours=discretization_unit)
        self.dataset = dataset
        self.window_n = 1
        self.prev_date = None
        self.next_date = None
        self.store_embs = np.array([])
        self.counts = 0

    def discretize_batch(self, timestamps_, features, step, n_batch):
        #with open(r'tfidf_WindowDataset1_3(1).csv', 'a', newline='') as csvfile:
        with open('temp2.mat','wb') as f:
            #fieldnames = ['Features X','Counts y']
            #fieldnames = ['Features '+ str(i) for i in range(features.shape[1])]
            #fieldnames.append('Counts y')
            #writer = csv.writer(csvfile, delimiter=",") #csv.DictWriter(csvfile, fieldnames=fieldnames)
            #writer.writerow(fieldnames)
            #Get timestamps
            timestamps = pd.to_datetime(timestamps_)

            if n_batch == 1:
                self.prev_date = timestamps[0]
                self.next_date = self.prev_date+self.delta 
                self.store_embs = np.array([])
            
            end_date = timestamps_[-1]
            n_ex = 0
            X = []
            y=[]
            while(1):  
                mask = np.logical_and(timestamps>=self.prev_date, timestamps<self.next_date)

                if not any(mask==True):
                    #dataset, window_n, store_embs, counts, prev_date, next_date = store(dataset, window_n, prev_date, next_date, delta, store_embs, counts, store_embs)
                    self.store(self.store_embs)
                    n_ex += 1
                    if n_ex == self.dataset['window_size']+1:
                        #self.sliding_window(writer)
                        tfidf, count = self.sliding_window(f)
                        X.append(tfidf)
                        y.append(count)
                        n_ex -= 1
                    continue

                self.counts += np.count_nonzero(mask)

                try:
                    aux = features[mask,:].mean(axis=0) #before: .todense()
                except:
                    print("error")

                if self.store_embs.size:
                    try:
                        avg_emb = np.average(np.vstack([self.store_embs, aux]), axis=0)#torch.cat((self.store_embs, aux), 0)#np.concatenate(store_embs, aux)
                    except:
                        print("error")
                else:
                    avg_emb = aux
                
                #if the last index is the date at the end of the batch, we need to open the next batch in order to
                #check if there are more input ids to store in the corresponding window
                if mask[-1] == True:#batch['timestamp'][mask][-1] == end_date:
                    assert np.array(timestamps_)[mask][-1]==end_date, "error"
                    self.store_embs = avg_emb
                    break

                #dataset, window_n, store_embs, counts, prev_date, next_date = store(dataset, window_n, prev_date, next_date, delta, avg_emb, counts, store_embs)
                self.store(avg_emb)
                n_ex += 1
                if n_ex == self.dataset['window_size']+1:
                    #self.sliding_window(writer)
                    tfidf, count = self.sliding_window(f)
                    if len(y)==0:
                        X = tfidf
                    else:
                        X = scipy.sparse.vstack([X, tfidf])
                    y.append(count)
                    n_ex -= 1
                
                if len(self.dataset['input_ids'])>self.dataset['window_size']+1:
                    print("memory issue!")
            #return dataset, window_n, prev_date, next_date, store_embs, counts
            savemat(f, {'X': X, 'y': y})

        
    def store(self, avg_emb):
        self.dataset['input_ids'].append({
            'id': self.window_n,
            'start_date': self.prev_date,
            'avg_emb': avg_emb,
            'count': self.counts,
        })
        self.window_n += 1

        self.store_embs = np.array([])
        self.counts = 0

        self.prev_date = self.next_date
        self.next_date = self.prev_date+self.delta

        #return dataset, window_n, store_embs, counts, prev_date, next_date

    def sliding_window(self, writer):
        #For each individual timestamp
        window_size = self.dataset['window_size']
        length = len(self.dataset['input_ids'])

        if window_size >= length:
            #logger.info("ERROR. WINDOW_SIZE IS TOO BIG! Loading next tweet batch...")
            return np.array([]), np.array([])
        else:

            #Calculates the timedifference
            timedif = [i for i in range(window_size)] 

            #Calculate the weights using K = 0.5 (giving 50% of importance to the most recent timestamp)
            #and tau = 6.25s so that when the temporal difference is 10s, the importance is +- 10.1%
            wi = weights(0.5, 2, timedif)
                                            
            idx = window_size

            while idx < length:
                start = self.dataset['input_ids'][idx]
        
                X = np.zeros(np.shape(self.dataset['input_ids'][0]['avg_emb']))
                for i in range(1,1+window_size):
                     X += wi[i-1]*self.dataset['input_ids'][idx-i]['avg_emb']

                idx += 1
            
            self.dataset['input_ids'] = self.dataset['input_ids'][len(X):]
            #del dataset
            #self.dataset['input_ids'] = []

            #writer.writerow({'Features X': np.array(X), 'Counts y': int(start['count'])})
            #X = np.squeeze(X).tolist()
            X = scipy.sparse.csr_matrix(np.squeeze(X))
            y = int(start['count'])
            #X.append(int(start['count']))
            #writer.writerow(X)
            #savemat(writer, {'y': int(start['count'])})   # append

            return X,y
            #yield X,y

'''This function receives the time difference vector and calculates the weight for each timestamp,
depending on the temporal distance to the most recent timestamp. It uses an exponential function.
The user can specify the constants K and tau of the exponential, that are inputs of the function.'''
def weights(k,tau,timedif):  
    length = len(timedif)
    
    #Creates the weight vector with all zeros, the length is the same as the time difference vector
    #(as many weights as there are timestamps)
    weight_vector = np.zeros(length)
    
    #For each timestamp, calculate the weight associated to it based on the following exponential:
    #K*e^(-Td/tau) where Td is the temporal distance to the most recent event, in seconds
    for i in range(0,length,1):
        weight_vector[i] = k*(np.exp((-timedif[i])/tau))
    
    #Normalize the weights so that the weight vector has unit norm
    if sum(weight_vector) != 1:
        weight_vector = weight_vector/sum(weight_vector)
        
    return weight_vector

def ChunkIterator(df, n_chunks, chunksize, n_tot_sent, n_en_sent, train_split_date, dev_split_date, test_split_date, window_size, discretization_unit):
    #tweet_times = []

    for df_chunk in df:
        print("Processing chunk n " + str(n_chunks+1))
        n_chunks += 1
        
        # Assert that the chunk is chronologically ordered
        #print("Checking if the chunk is chronologically ordered...")
        #df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], errors='ignore')

        #times = df_chunk['timestamp'].values
        #if all(times[:-1] <= times[1:]):
        #    print("OK")
        #else:
        #    print("NO") 
    
        #Each chunk is in df format
        if not df_chunk.empty:
        
            nRow, nCol = df_chunk.shape
            print('There are '+str(nRow)+' rows and '+str(nCol)+' columns')

            # For all tweets
            for tweet in range((n_chunks-1)*chunksize, nRow+(n_chunks-1)*chunksize, 1):
                tweet_time = pd.to_datetime(df_chunk['timestamp'][tweet])

                n_tot_sent += 1
                sentence = df_chunk['text'][tweet] #sentence/tweet to encode
                    
                if isinstance(sentence, str):
                    try:
                        _, _, details = cld2.detect(sentence)#detect(sentence)
                        language= details[0][1]
                    except:
                        language = 'none'
                        
                    # Only read english sentences
                    if language=='en': 
                        n_en_sent += 1

                        # CLEANING PHASE
                        sentence = get_cleaned_text(sentence)
                        if len(sentence)>0:
                            #tweet_times.append(tweet_time.value)   
                            if tweet_time < train_split_date:  
                                yield sentence,_,_
                            elif tweet_time >= train_split_date+datetime.timedelta(hours=window_size*discretization_unit) and tweet_time < dev_split_date:
                                yield _,sentence,_
                            elif tweet_time>= dev_split_date+datetime.timedelta(hours=window_size*discretization_unit):
                                yield _,_,sentence
                            
    #return tweet_times, n_en_sent, n_tot_sent 
          

def get_cleaned_text(sentence):
    return " ".join(twokenize.tokenize(sentence))#.encode('utf-8')

def load_data(path, chunks=False):
    # Load the bitcoin data
    #col_names = ["id", "user", "fullname", "url", "timestamp", "replies", "likes", "retweets", "text"]
    chunksize = 500000

    if chunks:
        df = pd.read_csv(path, delimiter=';',  engine='python', chunksize=chunksize)#, parse_dates=['timestamp'], index_col=['timestamp'])
    else:
        nRowsRead = None # specify 'None' if want to read whole file
        df = pd.read_csv(path, delimiter=';', nrows = nRowsRead)#, parse_dates=['timestamp'], index_col=['timestamp'])
    df.dataframeName = 'tweets.csv'

    return df

def save_dataframe(df):
    print("Saving dataframe in csv format...")
    df.to_csv(cpath.joinpath(r'bitcoin_data/sorted_tweets.csv'), index=False, sep=';')  #df.to_pickle("../bitcoin_data/bitcoin_df.pkl")  # where to save it, usually as a .pkl
    print("Done!")

def pre_process_df(df, timestamp=True):
    print("Removing rows containing nan values in timestamp or text columns...")

    # Remove rows from the dataframe if the timestamp or the text is not there, because we need both
    df = df.dropna(subset=['timestamp', 'text'])
    
    print("Done!")

    print("Ordering dataframe by timestamp...")

    # Organize dataframe by increasing order of timestamp (chronological)
    if timestamp:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='ignore')
    df = df.sort_values(by='timestamp', inplace=False)

    #Filter
    mask = (df['timestamp'] > '2019-05-01')
    df = df.loc[mask]

    print("Done!")

    return df

def main():
    parser = argparse.ArgumentParser(description='Create tf idfs of a tweet dataset to be used as embeddings.')
    
    parser.add_argument('--csv_path', default='bitcoin_data/', help="OS path to the folder where the input ids are located.")
    parser.add_argument('--discretization_unit', default=1, help="The discretization unit is the number of hours to discretize the time series data. E.g.: If the user choses 3, then one sample point will cointain 3 hours of data.")
    parser.add_argument('--window_size', default=3, help="Number of time windows to look behind. E.g.: If the user choses 3, when to provide the features for the current window, we average the embbedings of the tweets of the 3 previous windows.")
    parser.add_argument('--create', action="store_false", help="Do you want to create tf-idfs from a csv file or to load and create the dataset with windows?")
    args = parser.parse_args()
    print(args) 

    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    if args.create:
        # READ DATA
        # Find all .csv files in the bitcoin_data folder
        path = args.csv_path
        extension = 'csv'
        os.chdir(path)
        result = glob.glob('*.{}'.format(extension))

        if not result:
            print("No csv in the bitcoin_data folder!")
        else:
            # Check if there is any csv with the keyword 'sorted' in its name
            bool_sorted = [('sorted' in csv_file) for csv_file in result]
            bool_filtered = [('filtered' in csv_file) for csv_file in result]

            joint_bool = bool_sorted and bool_filtered

            if any(joint_bool): 
                print("Found sorted and filtered csv! Loading in chunks...")

                # Load data from sorted csv in chunks
                df = load_data(cpath.joinpath(r'bitcoin_data/'+ result[np.where(joint_bool)[0][0]]), chunks=True)

            else:
                print("Did not found sorted csv. Loading unsorted csv...")

                # Load data from csv
                df = load_data(cpath.joinpath(r'/bitcoin_data/tweets.csv'), chunks=False)

                df = pre_process_df(df)

                df["text"].groupby([df["timestamp"].dt.year, df["timestamp"].dt.month]).count().plot(kind="bar")
                plt.show()

                # Save the dataframe
                save_dataframe(df)

                
        #Read start date and end date
        path = cpath.joinpath(r'bitcoin_data/')
        extension = 'txt'
        os.chdir(path)
        result = glob.glob('*.{}'.format(extension))
        result = [file_ for file_ in result if re.match(r'ids_chunk[0-9]+',file_)] 
        result.sort(key=sortKeyFunc)
    
        result = [result[0], result[-1]]
        n = 1
        for f in result:
            with open(f, "r") as infile:
                data = json.load(infile)
                if n==1:
                    start_date = pd.to_datetime(list(map(int, data.keys()))[0])
                else:
                    end_date = pd.to_datetime(list(map(int, data.keys()))[-1])
                n += 1
                infile.close()

        print("Start date is "+str(start_date) + " and end date is "+ str(end_date) +".")

        time_delta = end_date-start_date

        #Total number of dt's
        n_dt = (time_delta.total_seconds()/(args.discretization_unit*3600))
        split_idx = np.cumsum(np.multiply(int(np.ceil(n_dt)),[0.8,0.1,0.1]))

        train_split_date = (start_date+datetime.timedelta(hours = split_idx[0])).tz_localize('US/Eastern')
        dev_split_date = (start_date+datetime.timedelta(hours = split_idx[1])).tz_localize('US/Eastern')
        test_split_date = (start_date+datetime.timedelta(hours = split_idx[2])).tz_localize('US/Eastern')

        n_chunks = 0
        n_en_sent = 0
        n_tot_sent = 0

        chunksize = 500000
                        
        #field_names = ['Sentence', 'Replies', 'Likes', 'Retweets', 'English']

        corpus  = ChunkIterator(df, n_chunks, chunksize, n_tot_sent, n_en_sent, train_split_date, dev_split_date, test_split_date, args.window_size, args.discretization_unit)
        tfidf = TfidfVectorizer(max_features=100000)
        train_corpus, dev_corpus, test_corpus = zip(*corpus)
        #Fit transform with train
        train_feature_matrix = tfidf.fit_transform(train_corpus)

        #Transform dev and test
        dev_feature_matrix = tfidf.transform(dev_corpus)
        test_feature_matrix = tfidf.transform(test_corpus)

        
        #tweet_times, n_en_sent, n_tot_sent = ChunkIterator(df, n_chunks, chunksize, n_tot_sent, n_en_sent)
        scipy.sparse.save_npz(r"C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\train_tfidf.npz", train_feature_matrix)
        scipy.sparse.save_npz(r"C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\dev_tfidf.npz", dev_feature_matrix)
        scipy.sparse.save_npz(r"C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\test_tfidf.npz", test_feature_matrix)

        # Percentage of english sentences
        print("Percentage of english sentences:"+ str((n_en_sent/n_tot_sent)*100) + " %.")

        #Saving
        #with open(r"C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\tweet_times.txt", "wb") as fp:   #Pickling
        #    pickle.dump(tweet_times, fp)

        #Save vectorizer: to analyze dictionary use .vocabulary_
        with open('vectorizer.pk', 'wb') as infile:
            pickle.dump(tfidf, infile)
    else:
        #Load tf-ids (features)
        sparse_matrix = scipy.sparse.load_npz(r"C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\tf_idf2.npz")

        
        #Load times
        with open(r"C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\tweet_times.txt", "rb") as fp:   # Unpickling
            tweet_times = pickle.load(fp)

        tweet_batch = TweetBatch(args.discretization_unit, args.window_size)

        tweet_batch.discretize_batch(tweet_times, sparse_matrix, 1, 1)
        #To average
        #sparse_matrix[0:100].todense().mean(axis=0).shape

        #Create dataset
        #Store dataset


    # End profiling code
    pr.disable()
    pr.print_stats(sort='time')

if __name__=="__main__":
    main()

