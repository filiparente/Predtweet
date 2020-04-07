import json
import pandas as pd

#Test how many points the new_cut_dataset has
filename = 'new_dataset.txt'
with open(filename, "r") as f:
        data = json.load(f)

        print(len(data['embeddings']))
        print(pd.to_datetime(data['start_date']))
        print(pd.to_datetime(data['end_date']))

        time_delta = pd.to_datetime(data['end_date'])-pd.to_datetime(data['start_date'])

        discretization_unit = 3

        #Total number of dt's
        n_dt = (time_delta.total_seconds()/(discretization_unit*3600))

        print(n_dt)