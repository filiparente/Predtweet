import os
import json
from datetime import timedelta
import pandas as pd
import argparse

def main():

    #Parser
    parser = argparse.ArgumentParser(description='Remove initial counts from the datasets because they are really small compared to the rest.')
    parser.add_argument('--dataset_path', default=".", help="Full path to the folder containing subfolders where the datasets are placed, for each combination of discretization unit and sliding window.")
    args = parser.parse_args()
    print(args)

    #select current directory
    directory = args.dataset_path

    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(directory)]                                                                            
    for subdir in subdirs:                                                                                            
        files = next(os.walk(subdir))[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(subdir + "/" + file)    
    r2 = [el for el in r if el.endswith('new_dataset.txt')]                                                                     

    #Get list of all folders in current datasets directory
    for filename in r2:
        print(filename)
        #for file in files:
        #    #print os.path.join(subdir, file)
        #    filepath = subdir + os.sep + file

        #    if filepath.endswith(".html"):
        #        print (filepath)

        #Get folder name
        #folder_name = folder[0]

        #if not folder_name in ['.', '..']:
        start_idx = 0
        
        #Load txt file containing the dataset
        with open(filename, "r") as f:
            data = json.load(f)

            for i in range(len(data['embeddings'])):
                #Cut-off index
                if data['embeddings'][i]['y']>1000:
                    start_idx = i
                    break
            
            #Cut dataset
            list_of_dicts = data['embeddings']
            data['embeddings'] = [list_of_dicts[i] for i in range(len(list_of_dicts)) if i>=start_idx]
            
            #Update start_date
            data['start_date'] = (pd.to_datetime(data['start_date'])+timedelta(hours=start_idx)).value

            #Store new dataset
            with open(filename[:-15]+'/new_cut_dataset.txt', "w") as f2:
                json.dump(data, f2)


            print("Done!")

if __name__== "__main__":
    main()