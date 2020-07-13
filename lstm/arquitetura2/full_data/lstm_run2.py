# 1. new_cut_dataset, num_train_epochs, deixar a correr com dataset grande, lstm com dw=3
# 1. deixar a correr com dataset grande, lstm so com dw=1
# 1. deixar a correr com dataset grande, lstm com media ponderada

import random
import torch
import numpy as np
#from datetime import timedelta
#import pandas as pd 
#from torch.nn.utils.rnn import pad_sequence
#import cProfile
#import progressbar
import os
import glob
#import re
#import datetime
import json
import argparse
import matplotlib.pyplot as plt
#import csv
#import matplotlib.pyplot as plt
#from pandas import read_csv
import math
from torch.utils.data import Dataset, DataLoader, SequentialSampler
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pdb
from scipy.io import loadmat
import scipy
import pandas as pd

torch.manual_seed(1)

# fix random seed for reproducibility
np.random.seed(7)


class MyCollator(object):
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __call__(self, batch):
        # do something with batch and self.params
        batch_size = self.batch_size
        seq_len = self.seq_len
        
        n_features = batch[0]['X'].shape[0]

        X = np.zeros((batch_size, seq_len, n_features))
        y = np.zeros((batch_size, seq_len))
        n = 0
        for i in range(batch_size):
            for j in range(seq_len):
                if n==len(batch): #exceeded dataset
                    if j<seq_len-1:
                        #remove last batch because it is not compleeted, and so we cannot concatenate it
                        X = X[:i,:,:]
                        y = y[:i, :]
                    return X,y
                X[i,j,:] = batch[n]['X']
                y[i,j] = batch[n]['y']
                n += 1

        #X, y = np.array([data[i]['X'] for i in range(len(data))]), np.array([data[i]['y'] for i in range(len(data))])#data[0]['X'], data[0]['y']  #isto e com batch_size=1 #create_dataset(data, look_back=window_size)
        return X, y #.float(),lengths.long()

class MyDataset(Dataset):
    def __init__(self, y, X, window_size, seq_len):
        self.dataset = []
        for i in range(len(y)):
            self.dataset.append({'X': X[:,i],'y': y[i]})

        assert len(self.dataset) == len(y), "error creating dataset"
            
        self.window_size = window_size
        self.seq_len = seq_len
        self.idx = 0
          
    def __len__(self):
        return len(self.dataset)

    def  __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #each feature is a concatenation of the average embeddings of the previous windows
        #X = [self.dataset[i]['X'] for i in range(self.idx,self.idx+self.seq_len)] 
        #sample = {'X': X, 'y': self.dataset[self.idx:self.idx+self.seq_len]['y'], 'window_size': self.window_size}

        #self.idx += self.seq_len

        #each feature is the average embedding in the current dt
        sample = {'X': self.dataset[idx]['X'], 'y': self.dataset[idx]['y'], 'window_size': self.window_size}

        return sample

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_dim, batch_size, device, num_layers=2):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers, dropout=0.2)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.hidden = None
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param,0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.hidden = None #(nn.Parameter(torch.randn(self.num_layers, batch_size, self.hidden_dim).type(torch.FloatTensor).to(device), requires_grad = True), nn.Parameter(torch.randn(self.num_layers, batch_size, self.hidden_dim).type(torch.FloatTensor).to(device), requires_grad=True)) #None
        self.output = None
        self.device = device
        #self.hidden = (nn.Parameter(torch.randn(self.num_layers, batch_size, self.hidden_dim).type(torch.FloatTensor), requires_grad=True), nn.Parameter(torch.randn(self.num_layers, batch_size, self.hidden_dim).type(torch.FloatTensor), requires_grad=True))

    def init_hidden(self, batch_size):
        #return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device), #hidden state
        #        torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)) #cell state
       return (nn.Parameter(torch.randn(self.num_layers, self.batch_size, self.hidden_dim).type(torch.FloatTensor).to(self.device), requires_grad=True), nn.Parameter(torch.randn(self.num_layers, self.batch_size, self.hidden_dim).type(torch.FloatTensor).to(self.device),requires_grad=True))

    def forward(self, inputs):
        # Push through RNN layer (the ouput is irrelevant)
        if inputs.shape[1] != self.hidden[0].shape[1]: #different batch sizes
            tuple_aux = (self.hidden[0][:,:inputs.shape[1],:].contiguous(), self.hidden[1][:,:inputs.shape[1],:].contiguous())
            self.hidden = None
            self.hidden = tuple_aux #BECAUSE TUPLES ARE IMMUTABLE

        self.output, self.hidden = self.lstm(inputs, self.hidden)
        return self.output, self.hidden

class Decoder(nn.Module):

    def __init__(self, hidden_dim, device, num_layers=2):
        super(Decoder, self).__init__()
        # input_size=1 since the output are single values
        #self.lstm = nn.LSTM(1, hidden_dim, num_layers=num_layers, dropout=0.2)
        self.out = nn.Linear(hidden_dim, 1)
        for name,param in self.out.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param,0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.device = device

    def forward(self, outputs, hidden, criterion):
        batch_size, seq_len = outputs.shape
        # Create initial start value/token
        #input = torch.tensor([[0.0]] * batch_size, dtype=torch.float).to(self.device)
        # Convert (batch_size, output_size) to (seq_len, batch_size, output_size)
        #input = input.unsqueeze(0)

        loss = 0
        preds = []
        #for i in range(seq_len):
            # Push current input through LSTM: (seq_len=1, batch_size, input_size=1)
            #output, hidden = self.lstm(input, hidden)
            # Push the output of last step through linear layer; returns (batch_size, 1)
            #output = self.out(output[-1])
        #output = self.out(hidden)
            # Generate input for next step by adding seq_len dimension (see above)
            #input = output.unsqueeze(0)
            # Compute loss between predicted value and true value
        #if outputs.shape[1]!=1:
        #    outputs = outputs.unsqueeze(0)

            #loss += criterion(output, outputs[:, i].view(-1,1))
        #loss = criterion(output.view(-1,1), outputs.view(-1,1))

        for i in range(seq_len):
            for j in range(batch_size):
                output = self.out(hidden[i,j,:])
                loss += criterion(output.view(-1,1), outputs[j,i].view(-1,1))
                preds.append(output)
        return loss, torch.cat(preds)
        #return loss, output

def set_seed(args,n_gpu):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = [dataset[i:(i+look_back)][j]['X'] for j in range(look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back]['y'])

    return np.array(dataX), np.array(dataY)

def evaluate(args, encoder, decoder, eval_dataloader, criterion, device, global_step, epoch, prefix="", store=True):
    #pdb.set_trace()
    # Validation
    eval_output_dir = args.output_dir
    results = {} 
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    
    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    num_eval_examples = len(eval_dataloader)
    print("  Num examples = %d", num_eval_examples)
    print("  Batch size = %d", 1)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    val_obs_seq = []
    val_preds_seq = []


    n_batch = 1

    eval_iterator = tqdm(eval_dataloader, desc="Evaluating")
     
    for step, batch in enumerate(eval_iterator):
        # Set our model to evaluation mode (as opposed to training mode) to evaluate loss on validation set
        encoder = encoder.eval()   
        decoder = decoder.eval()      

        trainX_sample, trainY_sample = batch
        
        if store:
            val_obs_seq.append(trainY_sample)

        n_batch += 1

        # Get our inputs ready for the network, that is, turn them into tensors
        trainX_sample = torch.tensor(trainX_sample, dtype=torch.float).to(device)
        #if trainX_sample.shape[0] == 1:
        #    trainX_sample = trainX_sample.unsqueeze(0)
        trainY_sample = torch.tensor(trainY_sample, dtype=torch.float).to(device)
        if trainX_sample.shape[0]==0: #no example
            continue
        # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
        trainX_sample = trainX_sample.transpose(1,0)
        
        # Run our forward pass
        #scores = model(trainX_sample)
        with torch.no_grad(): #in evaluation we tell the model not to compute or store gradients, saving memory and speeding up validation
            # Reset hidden state of encoder for current batch
            # STATELESS LSTM:
            #if encoder.hidden is None: #step==0:
            #    encoder.hidden = encoder.init_hidden(trainX_sample.shape[1])

            # Do forward pass through encoder: get hidden state
            #hidden = encoder(trainX_sample)    
            output, hidden = encoder(trainX_sample)
            # Compute the loss
            # Do forward pass through decoder (decoder gets hidden state from encoder)
            #tmp_eval_loss, logits = decoder(trainY_sample, hidden, criterion)
            tmp_eval_loss, logits = decoder(trainY_sample, output, criterion)

            val_preds_seq.append(logits)

            #print("Logits " + str(logits))
            #print("Counts " + str(trainY_sample))
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        trainY_sample = trainY_sample.detach().cpu().numpy()

        # Forward pass
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = trainY_sample
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, trainY_sample, axis=0)

    eval_loss = eval_loss / nb_eval_steps #it is the same as the mse! repeated code... but oh well!

    #preds = np.squeeze(preds) #because we are doing regression, otherwise it would be np.argmax(preds, axis=1)

    #since we are doing regression, our metric will be the mse
    result = {"mse":mean_squared_error(preds.ravel(), out_label_ids.ravel())} #compute_metrics(eval_task, preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    if not os.path.exists(output_eval_file):
        with open(output_eval_file, "w") as writer:
            print("***** Eval results {} in Global step {} and Epoch {}*****".format(prefix, global_step, epoch))
            writer.write("***** Eval results {} in Global step {} and Epoch {}*****\n".format(prefix, global_step, epoch))
            for key in sorted(result.keys()):
                print("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    else:
        with open(output_eval_file, "a") as writer:
            print("***** Eval results {} in Global step {} and Epoch {}*****".format(prefix, global_step, epoch))
            writer.write("***** Eval results {} in Global step {} and Epoch {}*****\n".format(prefix, global_step, epoch))
            for key in sorted(result.keys()):
                print("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results, val_obs_seq, val_preds_seq

def main():
    #Parser
    parser = argparse.ArgumentParser(description='Fit an LSTM to the data, to predict the tweet counts from the embbedings.')
    #C:/Users/Filipa/Desktop/Predtweet/bitcoin_data/datasets/dt/1.0/"
    parser.add_argument('--full_dataset_path', default=r"C:/Users/Filipa/Desktop/Predtweet/bitcoin_data/datasets/server/1.0/", help="OS path to the folder where the embeddings are located.")
    #r"C:/Users/Filipa/Desktop/Predtweet/bitcoin_data/TF-IDF/server/n_features/768/1.0/", help="OS path to the folder where the embeddings are located.")
    parser.add_argument('--discretization_unit', default=1, help="The discretization unit is the number of hours to discretize the time series data. E.g.: If the user choses 3, then one sample point will cointain 3 hours of data.")
    parser.add_argument('--window_size', type = int, default=0, help='The window length defines how many units of time to look behind when calculating the features of a given timestamp.')
    parser.add_argument('--seq_len', type = int, default=50, help='Input dimension (number of timestamps).')
    parser.add_argument('--batch_size', type = int, default=1, help='How many batches of sequence length inputs per iteration.')
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.") 
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for Adam.") #5e-5
    #parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    #parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    #parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Warmup is the proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%")
    parser.add_argument("--model_name_or_path", default=r'C:/Users/Filipa/Desktop/Predtweet/lstm/arquitetura2/sem_sliding_batches/lstm/fit_results/checkpoint-250/', type=str, help="Path to folder containing saved checkpoints, schedulers, models, etc.")    #r'C:/Users/Filipa/Desktop/Predtweet/bitcoin_data/TF-IDF/server/n_features/768/1.0/lstm/fit_results/checkpoint-200/', type=str, help="Path to folder containing saved checkpoints, schedulers, models, etc.")
    parser.add_argument("--output_dir", default='lstm/fit_results/', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_train_epochs", default=1500, type=int, help="Total number of training epochs to perform." )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--evaluate_during_training", action="store_false", help="Run evaluation during training at each logging step.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--test_acc", action="store_false", help="Run evaluation and store accuracy on test set.")
    parser.add_argument("--evaluate_only", action="store_true", help="Run only evaluation on validation and test sets with the best model found in training.")
    
    args = parser.parse_args()
    print(args)

    path = args.full_dataset_path
    discretization_unit = args.discretization_unit
    window_size = args.window_size
    seq_len = args.seq_len
    batch_size = args.batch_size
    train_dev_test_split = True
    normalize = True
    percentages = [0.8, 0.1, 0.1]

    # Load data
    os.chdir(path)
    files_ = glob.glob("*.mat")
    if len(files_)>1:
        #TF-IDF features: alphabetically order dev-test-train
        dev_data = loadmat(path+"dev_dataset.mat")
        test_data = loadmat(path+"test_dataset.mat")
        train_data = loadmat(path+"train_dataset.mat")
        
        data = dict()
        data['y'] = np.hstack(( train_data['y'][0],dev_data['y'][0],test_data['y'][0] )).ravel()
        data['X'] = scipy.sparse.vstack([train_data['X'],dev_data['X'], test_data['X']]).toarray()

        disc_unit = train_data['disc_unit'][0][0] #discretization unit in hours
        window_size = train_data['window_size'][0][0] #length of the window to average the tweets

        start_date = int(train_data['start_date'])
        end_date = int(test_data['end_date'])
        
        
        train_dev_test_split = False #already done!

    else:
        #BERT features 
        data = loadmat(path+'dataset.mat')
        
        start_date = data['start_date'][0][0]
        end_date = data['end_date'][0][0]
        
        disc_unit = data['disc_unit'][0][0] #discretization unit in hours
        window_size = data['window_size'][0][0] #length of the window to average the tweets

        train_dev_test_split = True

    pd_start_date = pd.to_datetime(start_date)
    
    pd_end_date = pd.to_datetime(end_date)
    print("Start date: " + str(pd_start_date) +" and end date: " + str(pd_end_date))

    obs_seq = data['y'].ravel() 
    X = np.transpose(data['X'])

    print("Observation sequence stats: min " + str(np.min(obs_seq)) + " max " + str(np.max(obs_seq)) + " mean " + str(np.mean(obs_seq)) + " std " + str(np.std(obs_seq)))


    plt.figure(figsize=(15,8))
    plt.title("Observation sequence before and after removing initial small counts")
    plt.xlabel("t")
    plt.ylabel("Observation sequence")
    plt.plot(obs_seq)

    #Cut initial small counts
    
    #Find first jump
    #idx = find((diff(obs_seq)./obs_seq(2:end))*100>90);   
    #idx = find(obs_seq>1000)
    
    diff_ = list(np.diff(obs_seq).ravel())
    diff_sorted = np.sort(diff_)
    values_diff_ = diff_sorted[-10:]
    idx2 = np.inf
    for i in range(10):
        idx_ = diff_.index(values_diff_[i])#[idx_,~] = find(values_diff_[i]==diff_)

        if idx_<idx2:
            idx2 = idx_
        
    #cut observation sequence and features (drop first samples that correspond
    #to small tweet counts
    obs_seq = obs_seq[idx2+1:]
    X = X[:, idx2+1:]

    plt.plot(obs_seq)
    #plt.show()       
            
    all_obs_seq = obs_seq
    all_X = X

    if train_dev_test_split:
        # Use train/dev/testtest split 80%10%10% to split our data into train and validation sets for training
        len_dataset = len(all_obs_seq)

        lengths = np.ceil(np.multiply(len_dataset,percentages))
        diff_ = int(np.sum(lengths)-len_dataset)
        if diff_>0:
            #subtract 1 starting from the end
            for i in range(len(lengths)-1,-1,-1):
                lengths[i] = lengths[i]-1
                diff_=diff_-1
                if diff_==0:
                   break
                

        lengths = np.cumsum(lengths)
        lengths = [int(l) for l in lengths]

        train_obs_seq = all_obs_seq[:lengths[0]]
        train_X = all_X[:, :lengths[0]]
        dev_obs_seq = all_obs_seq[lengths[0]+window_size:lengths[1]]
        dev_X = all_X[:, lengths[0]+window_size:lengths[1]]
        test_obs_seq = all_obs_seq[lengths[1]+window_size:]
        test_X = all_X[:, lengths[1]+window_size:]
    else:
        train_list = train_data['y'].ravel()
        c = list(train_list).index(obs_seq[0])
        train_obs_seq = np.transpose(train_list[c:])
        
        train_X = np.transpose(train_data['X'].todense()[c:,:])
        dev_obs_seq = np.transpose(dev_data['y'].ravel())
        dev_X = np.transpose(dev_data['X'].todense())    
        test_obs_seq = np.transpose(test_data['y'].ravel())
        test_X = np.transpose(test_data['X'].todense())
        
    print("Number of points in train dataset = " + str(len(train_obs_seq)))
    print("Number of points in dev dataset = " + str(len(dev_obs_seq)))
    print("Number of points in test dataset = " + str(len(test_obs_seq)))

    n_features = train_X.shape[0]
    train_mean = np.zeros((n_features,1))
    train_std = np.zeros((n_features,1))
    
    if normalize:
        # Normalization (z-score)
        for feature in range(n_features): #for all features, normalize independently for each feature
            # Get the z-score parameters from the training set (mean and std) 
            train_mean[feature] = np.mean(train_X[feature,:])
            train_std[feature] = np.std(train_X[feature,:])

            # Z-score the whole dataset with the parameters from training
            # z=(x-mean)/std
            train_X[feature,:]=(train_X[feature,:]-train_mean[feature])/train_std[feature]
            
            #min max scaling
            #maxV = max(train_X(feature,:));
            #minV = min(train_X(feature,:));
            #train_X(feature,:)   = (train_X(feature,:) - minV) / (maxV - minV);
            
            dev_X[feature,:]=(dev_X[feature,:]-train_mean[feature])/train_std[feature]
            test_X[feature,:]=(test_X[feature,:]-train_mean[feature])/train_std[feature]
        

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

   
    train_dataset = MyDataset(train_obs_seq, train_X, window_size, seq_len)
    dev_dataset = MyDataset(dev_obs_seq, dev_X, window_size, seq_len)
    test_dataset = MyDataset(test_obs_seq, test_X, window_size, seq_len)


    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory   
    my_collator = MyCollator(batch_size=1, seq_len=len(train_obs_seq))
    my_collator2 = MyCollator(batch_size=1, seq_len=len(dev_obs_seq))
    my_collator3 = MyCollator(batch_size=1, seq_len=len(test_obs_seq))

    print("Number of points in train dataset = " + str(len(train_dataset)))
    print("Number of points in dev dataset = " + str(len(dev_dataset)))
    print("Number of points in test dataset = " + str(len(test_dataset)))
    train_dataloader = DataLoader(train_dataset, sampler=SequentialSampler(train_dataset), batch_size=len(train_obs_seq), num_workers=1, collate_fn=my_collator)
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=len(dev_obs_seq), num_workers=1, collate_fn=my_collator2)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=len(test_obs_seq), num_workers=1, collate_fn=my_collator3)

    #if not os.path.isfile(args.output_dir+"/test_dataloader.pth"):
    #    torch.save(test_dataloader, args.output_dir+'/test_dataloader.pth')
    
    num_train_examples = len(train_dataloader)

    #Number of times the gradients are accumulated before a backward/update pass
    gradient_accumulation_steps = args.gradient_accumulation_steps

    # Number of training epochs, for finetuning in a specific NLP task, the authors recommend 2-4 epochs only
    epochs = args.num_train_epochs

    # input has dimension (samples, time steps, features)
    # create and fit the LSTM network
    EMBEDDING_DIM = 768 #number of features in data points
    HIDDEN_DIM = 128 #hidden dimension of the LSTM: number of nodes

    num_train_optimization_steps = int(num_train_examples/gradient_accumulation_steps)*epochs

    #model = LSTMRegression(trainX[0].shape, HIDDEN_DIM)
    #loss_function = nn.NLLLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    #with torch.no_grad():
    #    scores = model(trainX)
    #    print(scores)

    # If there's a GPU available...
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        n_gpu = 1

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    #Create model
    encoder = Encoder(EMBEDDING_DIM, HIDDEN_DIM, batch_size, device)
    decoder = Decoder(HIDDEN_DIM, device)

    #Put models in gpu
    encoder.cuda()
    decoder.cuda()

    # Create optimizers for encoder and decoder
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)

    # Check if saved optimizers exist    
    
    if os.path.isfile(os.path.join(args.model_name_or_path, "encoder_optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "decoder_optimizer.pt")
    ):
        # Load in optimizers' states
        encoder_optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "encoder_optimizer.pt")))
        decoder_optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "decoder_optimizer.pt")))
       
    #Regression criterion: mean squared error loss
    criterion = nn.MSELoss()
    global_step = 0
    epoch = 0
    
    if not args.evaluate_only:
        # Train!
        print("***** Running training *****")
        print("  Num examples = %d", num_train_examples)
        print("  Num Epochs = %d", epochs)
        print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        print("  Total optimization steps = %d", num_train_optimization_steps)

        # Store our loss and accuracy in the train set for plotting
        train_loss_set = []

        # Store our loss and accuracy in the validation set for plotting
        val_loss_set = []

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        best_mse_eval = np.inf
        save_best = False
        final_epoch = False
        first_eval = True
        best_val_preds_seq = []

        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path): #Path to pre-trained model
            # set global_step to global_step of last saved checkpoint from model path
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (num_train_examples // gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (num_train_examples // gradient_accumulation_steps)

            print("  Continuing training from checkpoint, will skip to saved global_step")
            print("  Continuing training from epoch %d", epochs_trained)
            print("  Continuing training from global step %d", global_step)
            print("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            
            #Load encoder and decoder states
            encoder.load_state_dict(torch.load(args.model_name_or_path+"encoder.pth"))
            decoder.load_state_dict(torch.load(args.model_name_or_path+"decoder.pth"))

            best_model_dir = args.model_name_or_path.rsplit("/",2)[0]+"/" #previous folder of the checkpoint (same as doing cd ..)
            if os.path.exists(best_model_dir+ "best_model/"):
                if os.path.isfile(best_model_dir+ "best_model/best_mse_eval.bin"):
                    #Load best_mse_eval
                    best_mse_eval = torch.load(best_model_dir+"best_model/best_mse_eval.bin")

                if os.path.isfile(best_model_dir+ "best_model/best_val_preds_seq.pt"):
                    #Load best_val_preds_seq
                    best_val_preds_seq = torch.load(best_model_dir+"best_model/best_val_preds_seq.pt")
            if os.path.isfile(os.path.join(args.model_name_or_path, "train_loss_set.pt")):
                train_loss_set = torch.load(args.model_name_or_path+"train_loss_set.pt")
            if os.path.isfile(os.path.join(args.model_name_or_path, "val_loss_set.pt")):
                val_loss_set = torch.load(args.model_name_or_path+"val_loss_set.pt")

        tr_loss, logging_loss = 0.0, 0.0
        n_eval = 1

        #model.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()

        # trange is a tqdm wrapper around the normal python range
        train_iterator = trange(
            epochs_trained, epochs, desc="Epoch",
        )

        set_seed(args, n_gpu)  # Added here for reproductibility

        #Training
        for epoch in train_iterator:
            if epoch == args.num_train_epochs-1:
                final_epoch = True
                train_obs_seq = []
                train_preds_seq = []


            epoch_iterator = tqdm(train_dataloader, desc="Iteration") 

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            n_batch = 1
            
            encoder.hidden = encoder.init_hidden(batch_size)

            # Train the data for one epoch
            for step, batch in enumerate(epoch_iterator): 
                # Set our model to training mode (as opposed to evaluation mode)
                encoder = encoder.train()
                decoder = decoder.train() 

                
                # Skip past any already trained steps if resuming training
                #if steps_trained_in_current_epoch > 0:
                #    steps_trained_in_current_epoch -= 1
                #    continue
                 
                trainX_sample, trainY_sample = batch
                if trainX_sample.shape[0]==0:#no example
                    continue
                if final_epoch:
                    train_obs_seq.append(trainY_sample)

                n_batch += 1

                # Get our inputs ready for the network, that is, turn them into tensors
                trainX_sample = torch.tensor(trainX_sample, dtype=torch.float).to(device)
                #if trainX_sample.shape[0] == 1:
                #    trainX_sample = trainX_sample.unsqueeze(0)
                trainY_sample = torch.tensor(trainY_sample, dtype=torch.float).to(device)
                
                # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
                trainX_sample = trainX_sample.transpose(1,0)
                #print(trainY_sample)
                # Run our forward pass
                #scores = model(trainX_sample)

                # Zero gradients of both optimizers (by default they accumulate)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                # Reset hidden state of encoder for current batch
                # STATELESS LSTM
                #if step==0:
                #    encoder.hidden = encoder.init_hidden(trainX_sample.shape[1])

                
                # Do forward pass through encoder: get hidden state
                #hidden = encoder(trainX_sample)    
                output, hidden = encoder(trainX_sample)

                # Compute the loss
                # Do forward pass through decoder (decoder gets hidden state from encoder)
                #loss, preds = decoder(trainY_sample, hidden, criterion)
                loss, preds = decoder(trainY_sample, output, criterion)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backpropagation, compute gradients 
                loss.backward(retain_graph=True)

                
                #Store 
                train_loss_set.append(loss.item())

                if final_epoch:
                    train_preds_seq.append(preds.view(-1,1))
            
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += trainX_sample.size(1) #b_input.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    #CLIP THE GRADIENTS?
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)

                    # Update parameters and take a step using the computed gradient
                    encoder_optimizer.step()
                    decoder_optimizer.step()

                    encoder.zero_grad()
                    decoder.zero_grad()
                    global_step += 1
                    #print("Global step nº: " + str(global_step))

                    if args.save_steps>0 and ((global_step%args.save_steps==0 or save_best) or (final_epoch and step==0)):
                        # Save model checkpoint
                        if save_best:
                            output_dir = os.path.join(args.output_dir, "best_model")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            torch.save(best_mse_eval, output_dir+"/best_mse_eval.bin")
                            torch.save(best_val_preds_seq, output_dir+"/best_val_preds_seq.pt")
                            torch.save(best_val_hidden_states, output_dir+"/best_val_hidden_states.pt")
                            save_best = False
                        else:
                            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        torch.save(encoder.state_dict(), output_dir+"/encoder.pth")
                        torch.save(decoder.state_dict(), output_dir+"/decoder.pth")

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))

                        if final_epoch: #save train_loss_set, val_loss_set
                            torch.save(train_loss_set, output_dir+"/train_loss_set.pt")
                            torch.save(val_loss_set, output_dir+"/val_loss_set.pt") 

                        print("Saving model checkpoint to %s", output_dir)

                        torch.save(encoder_optimizer.state_dict(), os.path.join(output_dir, "encoder_optimizer.pt"))
                        torch.save(decoder_optimizer.state_dict(), os.path.join(output_dir, "decoder_optimizer.pt"))
                        print("Saving optimizers' states to %s", output_dir)

                    # PARA CONFIRMAR SE OS PESOS ESTAVAM A SER ALTERADOS OU NAO: 199 e o peso W e 200 e o peso b (bias) da layer de linear de classificacao/regressao: WX+b
                    #if global_step%args.logging_steps==0:#step%args.logging_steps==0:
                    #    b = list(model.parameters())[199].clone()
                    #    b2 = list(model.parameters())[200].clone()

                    #    print("Check if the classifier layer weights are being updated:") #logger.info
                    #    print("Weight W: "+str(not torch.equal(a.data, b.data)))  #logger.info
                    #    print("Bias b: " + str(not torch.equal(a2.data, b2.data))) #logger.info
                
            #Evaluate at the end of the epoch
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.logging_steps>0 and epoch%args.logging_steps==0: 
                    if nb_tr_steps == 0:
                        nb_tr_steps =1
                    print("Train loss : {}".format(tr_loss/nb_tr_steps))
                    logs={}
                        
                    # DUVIDA: Pedir a zita para explicar isto
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    logs["loss"] = str(loss_scalar)
                    logging_loss = tr_loss

                    if args.evaluate_during_training: 
                        if first_eval: #first evaluation: store validation observation sequence, do not need to overwrite it because it is fixed
                            results, val_obs_seq, val_preds_seq = evaluate(args, encoder, decoder, dev_dataloader, criterion, device, global_step, epoch, prefix = str(n_eval), store=True)

                            first_eval = False
                        else:
                            results, _, val_preds_seq = evaluate(args, encoder, decoder, dev_dataloader, criterion, device, global_step, epoch, prefix = str(n_eval), store=False)
                            
                        n_eval += 1
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = str(value)

                        if results["mse"] < best_mse_eval:
                            save_best = True
                            best_mse_eval = results["mse"]
                            best_val_preds_seq = val_preds_seq
                            best_val_hidden_states = encoder.hidden
                            
                        #Store 
                        val_loss_set.append((results["mse"], global_step, epoch)) 


                        print(json.dumps({**logs, **{"step": str(global_step)}}))


                #print("Loss:", loss.item())

                #loss = loss_function(scores, trainY_sample)
                #loss.backward()
                #optimizer.step()


                if args.max_steps > 0 and global_step > args.max_steps:
                        epoch_iterator.close()
                        break
                
            
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        # Plot training loss (mse)
        plt.figure(figsize=(15,8))
        plt.title("Training loss com batch size "+str(batch_size)+ " and sequence length " + str(seq_len))
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.plot(train_loss_set)
        #plt.show()

        plt.savefig(os.path.join(args.output_dir)+'training_loss.png', bbox_inches='tight')

        # Plot validation loss (mse)
        plt.figure(figsize=(15,8))
        plt.title("Validation loss with batch size "+str(batch_size)+" and sequence length "+str(seq_len))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot([p[2] for p in val_loss_set], [p[0] for p in val_loss_set])
        #plt.show()

        plt.savefig(os.path.join(args.output_dir)+'validation_loss.png', bbox_inches='tight')

        # Plot true observations and predictions in the end of the training process (final epoch), in the same figure
        plt.figure(figsize=(15,8))
        plt.title("Train observation sequence: real and predicted")
        plt.xlabel("Sample")
        plt.ylabel("Count")
        obs_plot, = plt.plot(np.concatenate(train_obs_seq).ravel().tolist(), color='blue', label='Train observation sequence (real)')
        pred_plot, = plt.plot(torch.cat(train_preds_seq).detach().cpu().numpy().ravel().tolist(), color='orange', label='Train observation sequence (predicted)')
        plt.legend(handles=[obs_plot, pred_plot])
        #plt.show()

        plt.savefig(os.path.join(args.output_dir)+'train_obs_preds_seq.png', bbox_inches='tight')


        # Plot true observations and predictions in the validation set using the best model (the one that achieved the lowest mse in the validation set), in the same figure
        plt.figure(figsize=(15,8))
        plt.title("Validation observation sequence: real and predicted")
        plt.xlabel("Sample")
        plt.ylabel("Count")
        obs_plot, = plt.plot(np.concatenate(val_obs_seq).ravel().tolist(), color='blue', label='Train observation sequence (real)')
        pred_plot, = plt.plot(torch.cat(best_val_preds_seq).detach().cpu().numpy().ravel().tolist(), color='orange', label='Train observation sequence (predicted)')
        plt.legend(handles=[obs_plot, pred_plot])
        #plt.show()

        plt.savefig(os.path.join(args.output_dir)+'best_val_obs_preds_seq.png', bbox_inches='tight')

    # Check accuracy in test set

    # Load best model
    if args.test_acc:
        best_model_dir = os.path.join(args.output_dir, "best_model/")
        if os.path.exists(best_model_dir): #Path to best model (the one which gave lower MSE in the validation set during training)
            #Create encoder and decoder models
            best_encoder = Encoder(EMBEDDING_DIM, HIDDEN_DIM, batch_size, device)
            best_decoder = Decoder(HIDDEN_DIM, device)

            #Put models in gpu
            best_encoder.cuda()
            best_decoder.cuda()
         
            
            #Load encoder and decoder states
            best_encoder.load_state_dict(torch.load(best_model_dir+"encoder.pth"))  
            best_decoder.load_state_dict(torch.load(best_model_dir+"decoder.pth"))
            #encoder.hidden = torch.load(best_model_dir+"best_val_hidden_states.pt") 
            best_encoder.hidden = torch.load(best_model_dir+"best_val_hidden_states.pt")
            results, test_obs_seq, test_preds_seq = evaluate(args, best_encoder, best_decoder, test_dataloader, criterion, device, global_step, epoch, prefix = 'Test', store=True)          
            logs = {}
            for key, value in results.items():
                eval_key = "eval_{}".format(key)
                logs[eval_key] = str(value)

    # Plot true observations and predictions in the test set using the best model (the one that achieved the lowest mse in the validation set), in the same figure
    plt.figure(figsize=(15,8))
    plt.title("Test observation sequence: real and predicted")
    plt.xlabel("Sample")
    plt.ylabel("Count")
    obs_plot, = plt.plot(np.concatenate(test_obs_seq).ravel().tolist(), color='blue', label='Train observation sequence (real)')
    pred_plot, = plt.plot(torch.cat(test_preds_seq).detach().cpu().numpy().ravel().tolist(), color='orange', label='Train observation sequence (predicted)')
    plt.legend(handles=[obs_plot, pred_plot])
    #plt.show()

    plt.savefig(os.path.join(args.output_dir)+'test_obs_preds_seq.png', bbox_inches='tight')

    # See what the scores are after training
    #with torch.no_grad():
    #    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    #    tag_scores = model(inputs)
    #    print(tag_scores)
    #model = Sequential()
    #model.add(LSTM(4, input_shape=(look_back, 1)))
    #model.add(Dense(1))
    #model.compile(loss='mean_squared_error', optimizer='adam')
    #model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    
    print("Done!")
	


if __name__=='__main__':
    main()
