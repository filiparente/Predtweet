import numpy as np
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
'exec(%matplotlib inline)'

def load_dataset(path):
        with open(path,encoding='utf-8', errors='ignore', mode='r') as j:
            data = json.loads(j.read())

            features = [el['X'] for el in data['embeddings']]
            labels = [el['y'] for el in data['embeddings']]

            window_size = data['window_size']
            disc_unit = data['disc_unit']

            return features, labels, window_size, disc_unit

def main():
    parser = argparse.ArgumentParser(description='Plot the tweets distribution.')
    
    parser.add_argument('--path', default='results/', help="OS path to the folder where the dataset is located.")
    
    args = parser.parse_args()
    print(args) 

    #current path
    cpath = Path.cwd()

    path = cpath.joinpath(args.path)

    folders = [x[0] for x in os.walk(path)]
    folders = folders[1:]

    for i in range(len(folders)):
        plot_tweet_dist(folders[i]+"/dataset.txt", show=False)
        plt.savefig(folders[i]+'/tweets_distribution.png')
    #plt.show()

def plot_tweet_dist(filepath, show=True):
    X, y, window_size, disc_unit = load_dataset(filepath)

    plt.figure(figsize=(15,8))
    plt.title("Tweets distribution, window size="+str(window_size))
    plt.xlabel("Time")
    plt.ylabel("Number of tweets within "+str(disc_unit)+" hours")
    plt.plot(y)
    if show:
        plt.show()
    

if __name__=="__main__":
    main()