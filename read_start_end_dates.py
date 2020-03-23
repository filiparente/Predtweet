import json
import pandas as pd
import pdb

# Read json
path="ids_chunk1.txt"
with open(path,encoding='utf-8', errors='ignore', mode='r') as j:
    data = json.loads(j.read())
    print("done!")
    pdb.set_trace()
    timestamps_ = list(map(int, data['timestamps']))
    timestamps = pd.to_datetime(timestamps_)

    print("Start date is : " + str(timestamps[0]) + " and end date is : " +str(timestamps[-1])) 