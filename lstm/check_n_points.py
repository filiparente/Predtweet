import json
import pandas as pd
import argparse
#Test how many points the new_cut_dataset has
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default="new_dataset.txt", type=str, help="Full path to the txt file containing the dataset")
parser.add_argument('--discretization_unit', default=1, type=int, help="Unit of discretization in hours")

args = parser.parse_args()

filename = args.dataset_path
discretization_unit = args.discretization_unit

with open(filename, "r") as f:
        data = json.load(f)

        print(len(data['embeddings']))
        print(pd.to_datetime(data['start_date']))
        print(pd.to_datetime(data['end_date']))

        time_delta = pd.to_datetime(data['end_date'])-pd.to_datetime(data['start_date'])

        #Total number of dt's
        n_dt = (time_delta.total_seconds()/(discretization_unit*3600))

        print(n_dt)

