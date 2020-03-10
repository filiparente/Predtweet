import json
import glob
import os
from pathlib import Path
import torch
import re 
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import numpy as np
import itertools
import math
import pdb

MAX_LEN = 512

class MyDataset(Dataset):
    def __init__(self, json_filename):
        self.json_filename = json_filename
        with open(self.json_filename, "r") as infile:
            self.data = json.load(infile)
            infile.close()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        timestamp = list(map(int, self.data.keys()))[idx]
        input_ids = np.array(self.data[str(timestamp)][0], dtype=int)

        # Pad to the max length so that all samples have the same dimension
        input_ids = np.concatenate((input_ids, np.zeros(MAX_LEN-len(input_ids), dtype=int)), axis=None)

        # Create a mask of 1s for each token followed by 0s for padding
        seq_mask = [float(i>0) for i in input_ids]

        sample = {'timestamp': timestamp, 'input_ids': input_ids, 'attention_mask': seq_mask}

        return sample
class MyDataset2(Dataset):
    def __init__(self, data):
        self.data=data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        
        timestamp = list(map(int, self.data.keys()))[idx]
        input_ids = np.array(self.data[str(timestamp)][0], dtype=int)

        # Pad to the max length so that all samples have the same dimension
        #input_ids = np.concatenate((input_ids, np.zeros(MAX_LEN-len(input_ids), dtype=int)), axis=None)

        # Create a mask of 1s for each token followed by 0s for padding
        #seq_mask = [float(i>0) for i in input_ids]

        #sample = {'timestamp': timestamp, 'input_ids': input_ids, 'attention_mask': seq_mask}
        #sample = {'timestamp': timestamp, 'input_ids': input_ids}

        #return sample
        return (timestamp, input_ids)
def pad_and_sort_batch(data_loader_batch):
    """
    data_loader_batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    tokens, tags, lengths = tuple(zip(*data_loader_batch))
    x = pad_sequence(tokens, batch_first=True, padding_value=0)
    y = pad_sequence(tags, batch_first=True, padding_value=0)
    lengths, perm_idx = torch.IntTensor(lengths).sort(0, descending=True)
    return x[perm_idx, ...], y[perm_idx, ...], lengths

def collate_fn_padd(batch):
    '''
    batch is a list of (sequence, target, length) tuples
    Padds batch of variable length and returns a tensor of sequences padded to the maximum length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
    ## padd
    batch = [ torch.Tensor(t).to(device) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0).to(device)
    return batch, lengths, mask

def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    #_, labels, lengths = zip(*data)
    timestamps, list_input_ids = zip(*data)
    n_examples = len(data)
    #lengths = [len(input_ids) for input_ids in list_input_ids]
    max_len = MAX_LEN #max(lengths)
    features = torch.zeros((n_examples, max_len), dtype=int)
    #lengths = torch.tensor(lengths)

    for i in range(n_examples):
        features[i] = torch.cat([torch.tensor(list_input_ids[i]), torch.IntTensor(max_len-len(list_input_ids[i])).zero_()])

    return timestamps, features #.float(),lengths.long()

def random_split_ConcatDataset(ds, lengths, skip):
    """
    Roughly split a Concatdataset into non-overlapping new datasets of given lengths.
    Samples inside Concatdataset should already be shuffled

    :param ds: Dataset to be split
    :type ds: Dataset
    :param lengths: lengths of splits to be produced
    :type lengths: list
    :param skip: Number of samples to skip so that train and dev do not contain repeated information (this is mandatory due to the fact that we use sliding windows for past events)
    """
    if sum(lengths) != len(ds):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    idx_dataset = np.where(np.array(ds.cumulative_sizes) > lengths[0])[0][0]
    idx_dataset2 = np.where(np.array(ds.cumulative_sizes) > lengths[0]+ lengths[1])[0][0]
    assert idx_dataset >= 1, "Dev_split ratio is too large, there is no data in train set. " \
                            f"Please lower dev_split = {self.processor.dev_split}"

    split_idx = lengths[0]-ds.cummulative_sizes[idx_dataset-1]
    extra_dataset = MyDataset2({key: ds.datasets[idx_dataset].data[key] for key in list(ds.datasets[idx_dataset].data.keys())[:split_idx]})
    list_ = []
    for i in range(idx_dataset):
        list_.append(MyDataset2(ds.datasets[i].data))
    list_.append(extra_dataset)

    train = data.ConcatDataset(list_) # OU  train = data.ConcatDataset([ds.datasets[:idx_dataset-1], extra_dataset])
    
    split_idx += skip
    if split_idx > len(ds.datasets[idx_dataset]):
        pass
    else:
        extra_dataset = MyDataset2({key: ds.datasets[idx_dataset].data[key] for key in list(ds.datasets[idx_dataset].data.keys())[split_idx:]})
        list_ = []
        list_.append(extra_dataset)
        for i in range(idx_dataset+1, idx_dataset2):#len(ds.cummulative_sizes)):
            list_.append(MyDataset2(ds.datasets[i].data))

        split_idx = (lengths[0]+lengths[1])-ds.cummulative_sizes[idx_dataset2-1]
        extra_dataset = MyDataset2({key: ds.datasets[idx_dataset2].data[key] for key in list(ds.datasets[idx_dataset2].data.keys())[:split_idx]})

        list_.append(extra_dataset)

        dev = data.ConcatDataset(list_) # OU dev = data.ConcatDataset([extra_dataset, ds.datasets[idx_dataset:]])
    split_idx += skip
    if split_idx > len(ds.datasets[idx_dataset2]):
        pass
    else:
        extra_dataset = MyDataset2({key: ds.datasets[idx_dataset2].data[key] for key in list(ds.datasets[idx_dataset2].data.keys())[split_idx:]})
        list_ = []
        list_.append(extra_dataset)
        for i in range(idx_dataset2+1, len(ds.cummulative_sizes)):
            list_.append(MyDataset2(ds.datasets[i].data))
      
        test = data.ConcatDataset(list_) # OU dev = data.ConcatDataset([extra_dataset, ds.datasets[idx_dataset:]])
    
    assert  len(ds)-(train.cummulative_sizes[-1]+dev.cummulative_sizes[-1]+test.cummulative_sizes[-1])==2*skip, "ERROR: Sum of lengths of train, dev and test does not equal the length of the input dataset (minus the 2 window skips between train, test and dev)"
    
    return train, dev, test

#This function returns the numeric part of the file name and converts to an integer
def sortKeyFunc(s):
    return int(os.path.basename(s)[9:-4])

def load_inputids(path = "bitcoin_data/", batch_size=1000, window_size=3, discretization_unit=1, approx=((3600*10)/15)):
    
    extension = 'txt'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    result = [file_ for file_ in result if re.match(r'ids_chunk[0-9]+',file_)] 
    result.sort(key=sortKeyFunc)
    result=[result[0]]


    print("Files of chunks being analyzed: " + str(result))
    pdb.set_trace()

    if len(result)>1:
        list_of_datasets = []
        for f in result:
            list_of_datasets.append(MyDataset(f))

        # once all single json datasets are created you can concat them into a single one:
        multiple_json_dataset = data.ConcatDataset(list_of_datasets)
        length = multiple_json_dataset.cummulative_sizes[-1]
    
    else:
        dataset = MyDataset(result[0])
        length = len(dataset)

    # Use train_test_split to split our data into train and validation sets for training
    percentages = [0.8, 0.1, 0.1]
    lengths = [int(math.ceil(length*p)) for p in percentages]

    diff = int(sum(lengths)-length)
    if diff>0:
        #subtract 1 starting from the end
        for i in range(len(lengths)-1,-1,-1):
            lengths[i] = lengths[i]-1
            diff-=1
            if diff==0:
               break

    skip = round(window_size*discretization_unit*approx)
    
    if len(result)>1:
        train_dataset, dev_dataset, test_dataset = random_split_ConcatDataset(multiple_json_dataset, lengths, skip)
    else:
        train_dataset = MyDataset2({key: dataset.data[key] for key in list(dataset.data.keys())[:lengths[0]]})
        dev_dataset = MyDataset2({key: dataset.data[key] for key in list(dataset.data.keys())[lengths[0]+1:lengths[0]+1+lengths[1]]})
        test_dataset = MyDataset2({key: dataset.data[key] for key in list(dataset.data.keys())[lengths[0]+lengths[1]+2:]}) 
    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory
    train_dataloader = DataLoader(train_dataset, sampler=SequentialSampler(train_dataset), batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size, num_workers=4, collate_fn=collate_fn)

    #torch.save(train_dataloader, 'train_dataloader.pth')
    #torch.save(dev_dataloader, 'dev_dataloader.pth')
    #torch.save(test_dataloader, 'test_dataloader.pth')

    return train_dataloader, dev_dataloader, test_dataloader

   
if __name__=="__main__":
    #current path
    cpath = Path.cwd()

    inputids_path = cpath.joinpath("bitcoin_data/")

    dataloader = load_inputids(path=inputids_path)

    #TEST THAT THE MERGE IS GOOD
    iterator_ = iter(train_dataloader)
    for i in range(24001):
        batch = next(iterator_)
        if i==24000:
            print(batch)

    
    
