import os

if os.path.isfile(r'../bitcoin_data/bitcoin_df.pkl'):
    print("Pickle is present. Loading pickle...")
else:
    print("Pickle not present. Loading csv...")
