import torch
from transformers import BertModel, BertTokenizer, BertTokenizerFast
import numpy as np
import pandas as pd 
from torch.nn.utils.rnn import pad_sequence
import cProfile
import progressbar
from time import sleep
from sentence_transformers import SentenceTransformer
import os
import glob
import json
from langdetect import detect
import csv
import matplotlib.pyplot as plt
#from pyTweetCleaner import TweetCleaner
from preprocess import twokenize
from pathlib import Path

#current path
cpath = Path.cwd()

# Start profiling code
pr = cProfile.Profile()
pr.enable()

def get_cleaned_text(sentence):
    return " ".join(twokenize.tokenize(sentence)).encode('utf-8')

def load_data(path, chunks=False):
    # Load the bitcoin data
    col_names = ["id", "user", "fullname", "url", "timestamp", "replies", "likes", "retweets", "text"]
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

# READ DATA
# Find all .csv files in the bitcoin_data folder
path = 'bitcoin_data/'
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

model = SentenceTransformer('bert-base-nli-mean-tokens', device="cuda") #usar large?
#tc = TweetCleaner(remove_stop_words=False, remove_retweets=True)

n_chunks = 0
n_en_sent = 0
n_tot_sent = 0

chunksize = 500000
field_names = ['Sentence', 'Replies', 'Likes', 'Retweets', 'English']

for df_chunk in df:
    print("Processing chunk n " + str(n_chunks+1))
    n_chunks += 1

    # Assert that the chunk is chronologically ordered
    print("Checking if the chunk is chronologically ordered...")
    df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], errors='ignore')

    times = df_chunk['timestamp'].values
    if all(times[:-1] <= times[1:]):
        print("OK")
    else:
        print("NO") 
        # TODO

    # After loading the dataframe, view the first 10 lines
    df_chunk.head(10)


    #embbedings = {0: (0, [0,0])} #outra forma de fazer para nao dar type error la em baixo?
    #embbedings['tweet'] = []

    #Each chunk is in df format
    if not df_chunk.empty:
        embbedings  = {}
        #with open("language_detection_chunk" + str(n_chunks)+ ".csv", mode='w', encoding="utf-8") as csv_file:
           # writer = csv.DictWriter(csv_file, fieldnames=field_names)
           # writer.writeheader()

        nRow, nCol = df_chunk.shape
        print('There are '+str(nRow)+' rows and '+str(nCol)+' columns')

        # ENCODING PHASE
        print("\nENCODING PHASE\n")
        # `encode` will:
        #   (1) Tokenize the sentence (tweet).
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad the sequence to the maximum length (512)  -> indivual tokens.
        #   (6) The individual tokens for each word in the sentence (tweet) are fed to the model, 
        #       which is a BERT transformer fine-tuned to achieve better sentence encodings. It outputs the sentence embbedings.
                

        bar = progressbar.ProgressBar(maxval=nRow, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        # For all tweets
        for tweet in range((n_chunks-1)*chunksize, nRow+(n_chunks-1)*chunksize, 1):
            tweet_time = df_chunk['timestamp'][tweet]
                
            n_tot_sent += 1
            bar.update(tweet-(n_chunks-1)*chunksize+1)
            sentence = df_chunk['text'][tweet] #sentence/tweet to encode
                
            if isinstance(sentence, str):
                try:
                    language = detect(sentence)
                except:
                    language = 'none'
                    
                # Only read english sentences
                if language=='en': 
                    n_en_sent += 1

                    # CLEANING PHASE
                    #sentence = tc.get_cleaned_text(sentence)
                    sentence = get_cleaned_text(sentence)
                    if len(sentence)>0:

                        #writer.writerow({'Sentence': sentence, 'Replies': df_chunk['replies'][tweet], 'Likes': df_chunk['likes'][tweet], 'Retweets':df_chunk['retweets'][tweet],'English': 'YES'})
                        sentence_embedding = model.encode([sentence.decode('utf-8')])

                        #embbedings['tweet'].append({
                        #    'id': df_chunk['id'][tweet],
                        #    'timestamp': df_chunk['timestamp'][tweet],
                        #    'embbeding': sentence_embedding
                        #})
                        #embbedings[df_chunk['id'][tweet]] = (int(df_chunk['timestamp'].astype(np.int64)[tweet]), sentence_embedding)
                        embbedings[int(df_chunk['timestamp'].astype(np.int64)[tweet])] = sentence_embedding[0].tolist()
                        
                        #print(df_chunk['timestamp'][tweet])
               # else:
               #     writer.writerow({'Sentence': sentence, 'Replies': df_chunk['replies'][tweet], 'Likes': df_chunk['likes'][tweet], 'Retweets':df_chunk['retweets'][tweet], 'English': 'NO'})
                        
        bar.finish()

        if len(embbedings)>0:
            # Save json 
            with open("/mnt/hdd_disk2/frente/embbedings_chunk" + str(n_chunks) +".txt", 'w') as outfile:
                json.dump(embbedings, outfile)

        #csv_file.close()
          

# Percentage of english sentences
print("Percentage of english sentences:"+ str((n_en_sent/n_tot_sent)*100) + " %.")

# End profiling code
pr.disable()
pr.print_stats(sort='time')


# Read json
#json_data = json.loads("../bitcoin_data/embbedings_chunk" + str(n_chunks) +".txt")
