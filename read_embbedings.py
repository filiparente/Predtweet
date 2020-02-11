import os
import glob
#import simplejson as json
import json
import argparse
import pandas as pd
from datetime import timedelta
import numpy as np
import re
from pathlib import Path

#current path
cpath = Path.cwd()

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

'''This function receives a vector of timestamps and returns a vector of the time difference 
(in seconds, the maximum is 10 seconds because is the maximum window duration)
between each timestamp and the most recent timestamp in the vector.
For example: [14h55, 14h53, 14h52, 14h47] will return [0, 2, 3, 8]'''
def timedifference(timestamps):
    #Creates the time difference vector
    timediff_vec = []
    
    #Obtains the timestamp of the most recent event (always in the first index)
    t1 = timestamps[0]
    
    i = 0
    
    #Cycle the timestamps vector
    while i < len(timestamps):
        
        #Obtain the next timestamp
        timei = timestamps[i]
        
        #Calculate the (positive) difference to the most recent timestamp (t1)
        dif = t1-timei
        
        #Convert it to seconds
        dif = dif.seconds
        
        #Append the difference in the time difference vector
        timediff_vec.append(dif)
        
        i=i+1
        
    return timediff_vec

#TODO: make this a function which returns the dataset (X,y)

#This function returns the numeric part of the file name and converts to an integer
def sortKeyFunc(s):
    return int(os.path.basename(s)[16:-4])

def get_datasets(path, window_size, disc_unit):
    # disc_unit is no longer a string but a number corresponding to the number of hours
    #if disc_unit == 'min':
    #    delta = timedelta(minutes=1)
    #elif disc_unit == 'hour':
    #    delta = timedelta(hours=1)
    #elif disc_unit == 'day':
    #    delta = timedelta(days=1)
    #elif disc_unit == 'week':
    #    delta = timedelta(weeks=1)
    delta = timedelta(hours=disc_unit)

    # READ DATA
    # Find all json files in the bitcoin_data folder
    extension = 'txt'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    result = [file_ for file_ in result if re.match(r'embbedings_chunk[0-9]+',file_)] 
    result.sort(key=sortKeyFunc)
    print(result)

    dataset = {}
    dataset2 = {}
    dataset['disc_unit'] = disc_unit
    dataset['window_size'] = window_size
    dataset['embeddings'] = []

    if not result:
        print("No json files in the bitcoin_data folder!")
    else:
        n_file = 0
        window_n = 1
        counts = 0

        for json_file in result:
            print("Loading json "+json_file)

            # Read json
            json_file_path = path.joinpath(json_file)
            with open(json_file_path,encoding='utf-8', errors='ignore', mode='r') as j:
                data = json.loads(j.read())
                print("Done!")
                n_file += 1
                
                timestamps_ = list(map(int, data.keys()))
                timestamps = pd.to_datetime(timestamps_)

                if n_file == 1:
                    prev_date = timestamps[0]
                    next_date = prev_date+delta 
                    store_embs = np.array([])
                    dataset2['start_date'] = timestamps_[0]
               

                end_date = timestamps[-1]
                if n_file == len(result): #last file
                    dataset2['end_date'] = timestamps_[-1]

                while(1):
                
                    indexes = timestamps[np.logical_and(timestamps>=prev_date, timestamps<next_date)]

                    if len(indexes)==0:
                        dataset, window_n, store_embs, counts, prev_date, next_date = store(dataset, window_n, prev_date, next_date, delta, store_embs, counts, store_embs)
                        continue

                    counts += len(indexes)
                    nanoseconds = [str(int(round(index.timestamp()*1000000000))) for index in indexes]

                    try:
                        aux = np.average([data[idx] for idx in nanoseconds if len(data[idx])==768], axis=0)
                    except:
                        print("error")

                    if store_embs.size:
                        try:
                            avg_emb = np.average(np.vstack([store_embs, aux]), axis=0)
                        except:
                            print("error")
                    else:
                        avg_emb = aux
                    
                    #if the last index is the date at the end of the json file, we need to open the next json file in order to
                    #check if there are more embbedings to average in the corresponding window
                    if indexes[-1] == end_date:
                        store_embs = avg_emb
                        j.close()
                        break

                    dataset, window_n, store_embs, counts, prev_date, next_date = store(dataset, window_n, prev_date, next_date, delta, avg_emb, counts, store_embs)

    if store_embs.size:
        dataset['embeddings'].append({
            'id': window_n,
            'start_date': prev_date,
            'avg_emb': avg_emb,
            'count': counts,
        })          


    #For each individual timestamp
    if window_size>len(dataset['embeddings']):
        print("ERROR. WINDOW_SIZE IS TOO BIG!")
    else:
        #Calculates the timedifference
        timedif = [i for i in range(window_size)]
                            
       
        dataset2['disc_unit'] = disc_unit
        dataset2['window_size'] = window_size
        dataset2['embeddings'] = []
        #Calculate the weights using K = 0.5 (giving 50% of importance to the most recent timestamp)
        #and tau = 6.25s so that when the temporal difference is 10s, the importance is +- 10.1%
        wi = weights(0.5, 2, timedif)

        idx = window_size

        while idx < len(dataset['embeddings']):
            start = dataset['embeddings'][idx]
            
            X = np.zeros(np.shape(avg_emb))
            for i in range(1,1+window_size):
                X += wi[i-1]*dataset['embeddings'][idx-i]['avg_emb']

            dataset2['embeddings'].append({
                'X': X.tolist(),
                'y': start['count'],
            })
            idx += 1

        print("Done!")

    return dataset, dataset2

def store(dataset, window_n, prev_date, next_date, delta, avg_emb, counts, store_embs):
    dataset['embeddings'].append({
        'id': window_n,
        'start_date': prev_date,
        'avg_emb': avg_emb,
        'count': counts,
    })
    window_n += 1

    store_embs = np.array([])
    counts = 0

    prev_date = next_date
    next_date = prev_date+delta

    return dataset, window_n, store_embs, counts, prev_date, next_date

def save_dataset(base_path, dataset, dt, dw):
    #Creates folders in format dt.dw and dumps there the dataset.txt json file for each combination dt, dw
    #The folders are put inside the base_path provided by the user

    #TODO: how to save the dataset? as a json?
    # Save json 
    #print(path)
    #os.makedirs(os.path.dirname(path), exist_ok=True)
    path = "{}{}{}{}{}".format(base_path,'\\', dt, ".", dw)
    os.makedirs(path, exist_ok=True)
    with open(path + r'\dataset.txt', "w") as f:
        json.dump(dataset, f)
        f.close()

    
def main(window_size=5, disc_unit=1, embbedings_path="bitcoin_data/", out_path="results/"):
    embeddings_path = cpath.joinpath(embbedings_path)

    #dataset1 contains the embbedings within each discretization unit (since the start date) and the corresponding counts
    #dataset2 contains the weighted average of the window of previous embeddings, and the count for each discretization unit
    #The unit of discretization (minute, hour, day, week...) is chosen by the user
    #The length of the window is also chosen by the user, and it represents the number of previous embeddings to take into account
    dataset1, dataset2 = get_datasets(embeddings_path, window_size, disc_unit)

    if 'embeddings' in dataset2.keys(): #dt.dw combination out of range, cannot compute embeddings
        print("Dimension of the dataset: \n \t Nº of pairs (features/embbedings, labels/counts) = (X,y) = " + str(len(dataset2['embeddings'])))
        out_path = cpath.joinpath(out_path+'\\')
        save_dataset(out_path, dataset2, disc_unit, window_size)
    else:
        print("Not done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset from the sentence embbedings and timestamps.')
    
    parser.add_argument('--window_size', type = int, default=5, help='The window length defines how many units of time to look behind when calculating the features of a given timestamp.')
    parser.add_argument('--discretization_unit', type=int, default=1, help="Unit of time to discretize the time series data, as a string. Valid options are: 'min', 'hour', 'day', 'week'.")
    parser.add_argument('--embeddings_path', default='bitcoin_data/', help="OS path to the folder where the json files with the tweets embeddings are located.")
    parser.add_argument('--out_path', default='results/', help="OS path to the folder where the dataset must be saved.")
   

    args = parser.parse_args()
    print(args) 

    if isinstance(args.window_size, str):
        window_size = int(args.window_size)
    elif isinstance(args.window_size, int):
        window_size = args.window_size
    else:
        print("ERROR. Type of argument for input parameter window_size not understood.")

    if isinstance(args.discretization_unit, str):
        disc_unit = int(args.discretization_unit)
    elif isinstance(args.discretization_unit, int):
        disc_unit = args.discretization_unit
    else:
        print("ERROR. Type of argument for input parameter discretization_unit not understood.")

    main(window_size=window_size, disc_unit=disc_unit, embbedings_path=args.embeddings_path, out_path=args.out_path)

    

