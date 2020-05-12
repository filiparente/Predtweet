import limit_features as run
import itertools
import progressbar
import os
import argparse
import pdb
from dataclasses import dataclass
import glob

@dataclass
class Args:
    vectorizer_path: str
    n_features: float
    output_dir: str

parser = argparse.ArgumentParser(description='Create tf idfs of a tweet dataset to be used as embeddings.')
    
parser.add_argument('--dt_dir', default=r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\TF-IDF\dt\\', help="Full path where the train tfidf vectorizer is located")
parser.add_argument('--n_features', default=[2,3], help="Number of features to consider in the TfIdfVectorizer. Must be less than the number used during training.") #It can be a number or a vector
args1 = parser.parse_args()
    
print(args1) 
    
if isinstance(args1.n_features, int):
    n_features = [args1.n_features]
else:
    n_features = args1.n_features

path = args1.dt_dir
os.chdir(path)
result = glob.glob('*/')
#subfolders_path = [folder for folder in result if folder!='.' and folder!='..'] 
subfolders_path = [x[0] for x in os.walk(path) if len(x[1])!=0 and len(x[2])!=0]
subfolders_path.sort()
    
bar = progressbar.ProgressBar(maxval=len(subfolders_path)*len(n_features), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
args=Args("",0,"")
n=0

for subfolder_path in subfolders_path:
    subfolder_path = subfolder_path.replace('\\\\', '\\')
         
    bar.update(n+1)
    n+=1

    args.vectorizer_path = subfolder_path
    args.n_features = n_features
    args.output_dir = subfolder_path

    run.main(args)
    
bar.finish()
