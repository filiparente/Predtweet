# uncompyle6 version 3.7.2
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.16 |Anaconda, Inc.| (default, Sep 24 2019, 21:51:30) 
# [GCC 7.3.0]
# Embedded file name: /home/frente/Predtweet/read_embbedings.py
# Compiled at: 2020-03-25 18:18:19
import os, glob, json, argparse, pandas as pd
from datetime import timedelta
import numpy as np, re
from pathlib import Path
from scipy.io import savemat
import pdb
cpath = Path.cwd()

class TweetBatch:

    def __init__(self, discretization_unit, window_size):
        dataset = {}
        dataset['disc_unit'] = discretization_unit
        dataset['window_size'] = window_size
        dataset['embeds'] = []
        self.delta = timedelta(hours=discretization_unit)
        self.dataset = dataset
        self.window_n = 1
        self.prev_date = None
        self.next_date = None
        self.store_embs = np.array([])
        self.counts = 0
        self.X = []
        self.y = []
        self.n_ex = 0
        return

    def store(self, avg_emb):
        if self.dataset['window_size'] == 0:
            if len(self.y) == 0:
                self.X = avg_emb
            else:
                self.X = np.vstack([self.X, avg_emb])
            self.y.append(self.counts)
        self.dataset['embeds'].append({'id': self.window_n, 
           'start_date': self.prev_date, 
           'avg_emb': avg_emb, 
           'count': self.counts})
        self.window_n += 1
        self.store_embs = np.array([])
        self.counts = 0
        self.prev_date = self.next_date
        self.next_date = self.prev_date + self.delta

    def discretize_batch(self, timestamps_, data, step, n_file):
        timestamps = pd.to_datetime(timestamps_)
        if n_file == 1:
            self.prev_date = timestamps[0]
            self.next_date = self.prev_date + self.delta
            self.store_embs = np.array([])
        end_date = timestamps[(-1)]
        while 1:
            indexes = timestamps[np.logical_and(timestamps >= self.prev_date, timestamps < self.next_date)]
            if len(indexes) == 0:
                self.store(self.store_embs)
                self.n_ex += 1
                if self.dataset['window_size'] != 0 and self.n_ex == self.dataset['window_size'] + 1:
                    tfidf, count = self.sliding_window()
                    self.X = np.vstack([self.X, tfidf])
                    self.y.append(count)
                    self.n_ex -= 1
                continue
            self.counts += len(indexes)
            nanoseconds = [ str(int(round(index.timestamp() * 1000000000))) for index in indexes ]
            try:
                aux = np.average([ data[idx] for idx in nanoseconds if len(data[idx]) == 768 ], axis=0)
            except:
                print 'error'

            if self.store_embs.size:
                try:
                    avg_emb = np.average(np.vstack([self.store_embs, aux]), axis=0)
                except:
                    print 'error'

            else:
                avg_emb = aux
            if indexes[(-1)] == end_date:
                self.store_embs = avg_emb
                break
            self.store(avg_emb)
            self.n_ex += 1
            if self.dataset['window_size'] != 0 and self.n_ex == self.dataset['window_size'] + 1:
                tfidf, count = self.sliding_window()
                if len(self.y) == 0:
                    self.X = tfidf
                else:
                    self.X = np.vstack([self.X, tfidf])
                self.y.append(count)
                self.n_ex -= 1

    def sliding_window(self):
        window_size = self.dataset['window_size']
        length = len(self.dataset['embeds'])
        if window_size >= length:
            return (np.array([]), np.array([]))
        else:
            timedif = [ i for i in range(window_size) ]
            wi = weights(0.5, 2, timedif)
            idx = window_size
            while idx < length:
                start = self.dataset['embeds'][idx]
                X = np.zeros(np.shape(self.dataset['embeds'][0]['avg_emb']))
                for i in range(1, 1 + window_size):
                    X += wi[(i - 1)] * self.dataset['embeds'][(idx - i)]['avg_emb']

                idx += 1

            if all(np.squeeze(X) == X):
                skip = 1
            else:
                skip = np.shape(X)[0]
            self.dataset['embeds'] = self.dataset['embeds'][skip:]
            X = np.squeeze(X)
            y = int(start['count'])
            return (
             X, y)


def weights(k, tau, timedif):
    length = len(timedif)
    weight_vector = np.zeros(length)
    for i in range(0, length, 1):
        weight_vector[i] = k * np.exp(-timedif[i] / tau)

    if sum(weight_vector) != 1:
        weight_vector = weight_vector / sum(weight_vector)
    return weight_vector


def timedifference(timestamps):
    timediff_vec = []
    t1 = timestamps[0]
    i = 0
    while i < len(timestamps):
        timei = timestamps[i]
        dif = t1 - timei
        dif = dif.seconds
        timediff_vec.append(dif)
        i = i + 1

    return timediff_vec


def sortKeyFunc(s):
    return int(os.path.basename(s)[16:-4])


def get_datasets(path, window_size, disc_unit):
    extension = 'txt'
    os.chdir(path)
    result = glob.glob(('*.{}').format(extension))
    result = [ file_ for file_ in result if re.match('embbedings_chunk[0-9]+', file_) ]
    result.sort(key=sortKeyFunc)
    print result
    dataset = {}
    dataset2 = {}
    dataset2['disc_unit'] = disc_unit
    dataset2['window_size'] = window_size
    dataset2['embeddings'] = []
    if not result:
        print 'No json files in the bitcoin_data folder!'
    else:
        n_file = 0
        tweet_batch = TweetBatch(disc_unit, window_size)
        for json_file in result:
            print 'Loading json ' + json_file
            json_file_path = path.joinpath(json_file)
            with open(json_file_path, encoding='utf-8', errors='ignore', mode='r') as (j):
                data = json.loads(j.read())
                print 'Done!'
                n_file += 1
                timestamps_ = list(map(int, data.keys()))
                if n_file == 1:
                    dataset2['start_date'] = timestamps_[0]
                elif n_file == len(result):
                    dataset2['end_date'] = timestamps_[(-1)]
                tweet_batch.discretize_batch(timestamps_, data, 1, n_file)

        for i in range(len(tweet_batch.y)):
            dataset2['embeddings'].append({'X': list(tweet_batch.X[i, :]), 'y': tweet_batch.y[i]})

        dataset = tweet_batch.dataset
        return (
         dataset, dataset2, tweet_batch.X, tweet_batch.y)


def save_dataset(base_path, dataset, X, y, dt, dw):
    path = ('{}{}{}{}{}').format(base_path, '/', dt, '.', dw)
    os.makedirs(path, exist_ok=True)
    with open(path + '/new_dataset.txt', 'w') as (f):
        json.dump(dataset, f)
        f.close()
    with open(path + '/dataset.mat', 'wb') as (f):
        savemat(f, {'start_date': dataset['start_date'], 'end_date': dataset['end_date'], 'disc_unit': dt, 'window_size': dw, 'X': X, 'y': y})
        f.close()


def main(window_size=3, disc_unit=1, embbedings_path='bitcoin_data/', out_path='results/'):
    embeddings_path = cpath.joinpath(embbedings_path)
    dataset1, dataset2, X, y = get_datasets(embeddings_path, window_size, disc_unit)
    if 'embeddings' in dataset2.keys():
        print 'Dimension of the dataset: \n \t N of pairs (features/embbedings, labels/counts) = (X,y) = ' + str(len(dataset2['embeddings']))
        out_path = cpath.joinpath(out_path)
        save_dataset(out_path, dataset2, X, y, disc_unit, window_size)
    else:
        print 'Not done!'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset from the sentence embbedings and timestamps.')
    parser.add_argument('--window_size', type=int, default=3, help='The window length defines how many units of time to look behind when calculating the features of a given timestamp.')
    parser.add_argument('--discretization_unit', type=int, default=1, help="Unit of time to discretize the time series data, as a string. Valid options are: 'min', 'hour', 'day', 'week'.")
    parser.add_argument('--embeddings_path', default='bitcoin_data/', help='OS path to the folder where the json files with the tweets embeddings are located.')
    parser.add_argument('--out_path', default='results/', help='OS path to the folder where the dataset must be saved.')
    args = parser.parse_args()
    print args
    if isinstance(args.window_size, str):
        window_size = int(args.window_size)
    elif isinstance(args.window_size, int):
        window_size = args.window_size
    else:
        print 'ERROR. Type of argument for input parameter window_size not understood.'
    if isinstance(args.discretization_unit, str):
        disc_unit = int(args.discretization_unit)
    elif isinstance(args.discretization_unit, int):
        disc_unit = args.discretization_unit
    else:
        print 'ERROR. Type of argument for input parameter discretization_unit not understood.'
    main(window_size=window_size, disc_unit=disc_unit, embbedings_path=args.embeddings_path, out_path=args.out_path)
