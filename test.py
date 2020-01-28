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
import json
from langdetect import detect
import csv

# Start profiling code
pr = cProfile.Profile()
pr.enable()

def load_data(path):
    # Load the bitcoin data
    nRowsRead = None # specify 'None' if want to read whole file
    #col_names = ["id", "user", "fullname", "url", "timestamp", "replies", "likes", "retweets", "text"]

    df_chunk = pd.read_csv(path, delimiter=';', nrows = nRowsRead,  engine='python')
    df_chunk.dataframeName = 'tweets.csv'

    return df_chunk

def save_dataframe(df):
    print("Saving dataframe in pickle format...")
    df.to_pickle("../bitcoin_data/bitcoin_df.pkl")  # where to save it, usually as a .pkl
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
# Check if the pickle is already present
if os.path.isfile(r'../bitcoin_data/bitcoin_df.pkl'):
    print("Pickle is present. Loading pickle...")

    # Load data from pickle
    df = pd.read_pickle(r'../bitcoin_data/bitcoin_df.pkl')

    print("Pickle loaded!")

    # Assert that the dataframe contained in the pickle is chronologically ordered
    print("Checking if the dataframe is chronologically ordered...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='ignore')

    times = df['timestamp'].values
    if all(times[:-1] <= times[1:]):
        print("OK")
    else:
        print("NO")

        df = pre_process_df(df, timestamp=False)
    
        # Save the dataframe chronologically
        save_dataframe(df)

else:
    print("Pickle not present. Loading csv...")

    # Load data from csv
    df = load_data(r'../bitcoin_data/tweets.csv')

    df = pre_process_df(df)

    # Save the dataframe
    save_dataframe(df)

# After loading the dataframe, view the first 10 lines
df.head(10)

model = SentenceTransformer('bert-base-nli-mean-tokens')
embbedings  = {}
embbedings['tweet'] = []

#Each chunk is in df format
if not df.empty:
    with open("language_detection.csv", mode='w', encoding="utf-8") as csv_file:
        field_names = ['Sentence', 'English']
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()

        nRow, nCol = df.shape
        print(f'There are {nRow} rows and {nCol} columns')
        nRow = 10000
        # ENCODING PHASE
        print("\nENCODING PHASE\n")

        bar = progressbar.ProgressBar(maxval=nRow, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        # For all tweets
        for tweet in range(nRow):
            bar.update(tweet+1)
            sentence = df['text'][tweet] #sentence/tweet to encode
            
            if isinstance(sentence, str):
                try:
                    language = detect(sentence)
                except:
                    language = 'none'
                
                if language=='en':
                    writer.writerow({'Sentence': sentence, 'English': 'YES'})
                    # `encode` will:
                    #   (1) Tokenize the sentence (tweet).
                    #   (2) Prepend the `[CLS]` token to the start.
                    #   (3) Append the `[SEP]` token to the end.
                    #   (4) Map tokens to their IDs.
                    #   (5) Pad the sequence to the maximum length (512)  -> indivual tokens.
                    #   (6) The individual tokens for each word in the sentence (tweet) are fed to the model, 
                    #       which is a BERT transformer fine-tuned to achieve better sentence encodings. It outputs the sentence embbedings.
            
                    #sentence_embedding = model.encode([sentence])
                    #embbedings['tweet'].append({
                    #    'id': df['id'][tweet],
                    #    'timestamp': df['timestamp'][tweet],
                    #    'embbeding': sentence_embedding
                    #})
                else:
                    writer.writerow({'Sentence': sentence, 'English': 'NO'})
                    
        bar.finish()

# Save json 
#with open('../bitcoin_data/embbedings.txt', 'w') as outfile:
#    json.dump(embbedings, outfile)

# End profiling code
pr.disable()
pr.print_stats(sort='time')
