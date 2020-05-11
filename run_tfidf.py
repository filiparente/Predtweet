import create_tfidf_per_dt as run
import itertools
import progressbar
import os
import argparse
import pdb
from dataclasses import dataclass

@dataclass
class Args:
    discretization_unit: float
    window_size: float
    ids_path: str
    max_features: float
    output_dir: str
    create: bool
    csv_path:str

parser = argparse.ArgumentParser(description='Create a dataset from the sentence embbedings and timestamps.')
parser.add_argument('--csv_path', default=r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\\', help="OS path to the folder where the json files with the tweets embeddings are located.")
#parser.add_argument('--out_path', default='results/', help="OS path to the folder where the dataset must be saved.")
parser.add_argument('--output_dir', default=r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\dt', help="OS path to the folder where the dataset must be saved.")
parser.add_argument('--ids_path', default=r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\token_ids\\', help="Token ids path to read start date and end date from.")
parser.add_argument('--max_features', default=100000, help="Maximum number of features to consider in the TfIdfVectorizer.")
    
args1 = parser.parse_args()
print(args1) 

dt = [1,3,4,6,12,24,48]#[24, 48] #  #discretization unit in hours

dw = [0,1,3,5,7] #length of the sliding window of previous features, in units of dt

create=False

all_combs = list(itertools.product(dt, dw))

bar = progressbar.ProgressBar(maxval=len(all_combs), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
n_comb = 0
args=Args(0,0,"",0,"",False, "")

for comb in all_combs:
    dt = comb[0]
    dw = comb[1]
    bar.update(n_comb+1)
    n_comb+=1
    
    
    if dw==0:
        create=True
    print("Creating dataset for discretization unit "+str(dt)+" and window size "+str(dw)+"...")
    output_dir_dt = args1.output_dir + str(dt)+'/'#args.output_dir +'\\'+str(dt)+'\\'
    output_dir_dtdw = output_dir_dt + str(dt)+'.'+str(dw)+'/'#output_dir_dt + str(dt)+'.'+str(dw)+r'\\'

    if not os.path.exists(output_dir_dt):
        os.makedirs(output_dir_dt)
    
    if not os.path.exists(output_dir_dtdw):
        os.makedirs(output_dir_dtdw)
    
    args.discretization_unit = dt
    args.window_size = dw
    args.create = create
    args.output_dir = output_dir_dt
    args.ids_path = args1.ids_path
    args.max_features = args1.max_features
    args.csv_path = args1.csv_path
 
    run.main(args)
    if create: #create and store window 0
        create = False
        args.create=False
        run.main(args)
    
bar.finish()
