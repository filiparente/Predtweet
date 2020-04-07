# 1. new_cut_dataset, num_train_epochs, deixar a correr com dataset grande, lstm com dw=3
# 1. deixar a correr com dataset grande, lstm só com dw=1
# 1. deixar a correr com dataset grande, lstm com média ponderada

import random
import torch
import numpy as np
#from datetime import timedelta
#import pandas as pd 
#from torch.nn.utils.rnn import pad_sequence
#import cProfile
#import progressbar
import os
#import glob
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

torch.manual_seed(1)

# fix random seed for reproducibility
np.random.seed(7)


class MyDataset(Dataset):
    def __init__(self, json_filename, window_size, seq_len):
        self.json_filename = json_filename
        with open(self.json_filename, "r") as infile:
            self.data = json.load(infile) #json.loads(j.read())
            #extract dataset
            self.dataset = self.data['embeddings']
            

            infile.close()
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
        sample = {'X': [self.dataset[idx]['X']], 'y': self.dataset[idx]['y'], 'window_size': self.window_size}

        return sample

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_dim, device, num_layers=2):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers, dropout=0.2)
        self.hidden = None
        self.output = None
        self.device = device

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device), #hidden state
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)) #cell state

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
        self.device = device

    def forward(self, outputs, hidden, criterion):
        batch_size, seq_len = outputs.shape
        # Create initial start value/token
        #input = torch.tensor([[0.0]] * batch_size, dtype=torch.float).to(self.device)
        # Convert (batch_size, output_size) to (seq_len, batch_size, output_size)
        #input = input.unsqueeze(0)

        loss = 0
        #for i in range(seq_len):
            # Push current input through LSTM: (seq_len=1, batch_size, input_size=1)
            #output, hidden = self.lstm(input, hidden)
            # Push the output of last step through linear layer; returns (batch_size, 1)
            #output = self.out(output[-1])
        output = self.out(hidden)
            # Generate input for next step by adding seq_len dimension (see above)
            #input = output.unsqueeze(0)
            # Compute loss between predicted value and true value
        if outputs.shape[1]!=1:
            outputs = outputs.unsqueeze(0)

            #loss += criterion(output, outputs[:, i].view(-1,1))
        loss = criterion(output.view(-1,1), outputs.view(-1,1))
        return loss, output

class LSTMRegression(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMRegression, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to count space (a number, dim=1)
        self.hidden2count = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        count = self.hidden2count(lstm_out)
        return count

class MyCollator(object):
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __call__(self, batch):
        # do something with batch and self.params
        batch_size = self.batch_size
        seq_len = self.seq_len
        n_features = batch[0]['X'].shape[1]

        X = np.zeros((batch_size, seq_len, n_features))
        y = np.zeros((batch_size, seq_len))
        n = 1
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

        #X, y = np.array([data[i]['X'] for i in range(len(data))]), np.array([data[i]['y'] for i in range(len(data))])#data[0]['X'], data[0]['y']  #isto é com batch_size=1 #create_dataset(data, look_back=window_size)
        return X, y #.float(),lengths.long()

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

def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    window_size = data[0]['window_size']
    
    X, y = np.array([data[i]['X'] for i in range(len(data))]), np.array([data[i]['y'] for i in range(len(data))])#data[0]['X'], data[0]['y']  #isto é com batch_size=1 #create_dataset(data, look_back=window_size)
    return X, y #.float(),lengths.long()

def evaluate(args, encoder, decoder, eval_dataloader, criterion, device, global_step, epoch, prefix="", store=True):
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
        trainY_sample = torch.tensor(trainY_sample, dtype=torch.float).unsqueeze(0).to(device)


        # Convert (batch_size, seq_len, input_size) to (seq_len, batch_size, input_size)
        trainX_sample = trainX_sample.transpose(1,0)

        # Run our forward pass
        #scores = model(trainX_sample)
        with torch.no_grad(): #in evaluation we tell the model not to compute or store gradients, saving memory and speeding up validation
            # Reset hidden state of encoder for current batch
            # STATELESS LSTM:
            if step==0:
                encoder.hidden = encoder.init_hidden(trainX_sample.shape[1])

            # Do forward pass through encoder: get hidden state
            #hidden = encoder(trainX_sample)    
            output, hidden = encoder(trainX_sample)
            # Compute the loss
            # Do forward pass through decoder (decoder gets hidden state from encoder)
            #tmp_eval_loss, logits = decoder(trainY_sample, hidden, criterion)
            tmp_eval_loss, logits = decoder(trainY_sample, output, criterion)

            val_preds_seq.append(logits)

            print("Logits " + str(logits))
            print("Counts " + str(trainY_sample))
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

    preds = np.squeeze(preds) #because we are doing regression, otherwise it would be np.argmax(preds, axis=1)
    
    #since we are doing regression, our metric will be the mse
    result = {"mse":mean_squared_error(preds, out_label_ids)} #compute_metrics(eval_task, preds, out_label_ids)
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

def load_input(path = "bitcoin_data/", window_size=3, discretization_unit=1, seq_len=100, batch_size=3):
    
    dataset = MyDataset(path, window_size, seq_len)
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

    lengths = np.cumsum(lengths)
    train_dataset = [dataset[i] for i in range(lengths[0])]
    dev_dataset = [dataset[i] for i in range(lengths[0]+window_size, lengths[1])]
    test_dataset = [dataset[i] for i in range(lengths[1]+window_size, lengths[2]- window_size)]


    # normalize the dataset
    #scaler = MinMaxScaler(feature_range=(0, 1))

    #only the input features (X) need to be normalized
    X_train = np.array([train_dataset[i]['X'] for i in range(len(train_dataset))])
    X_dev = np.array([dev_dataset[i]['X'] for i in range(len(dev_dataset))])
    X_test = np.array([test_dataset[i]['X'] for i in range(len(test_dataset))])

    num_instances, num_time_steps, num_features = X_train.shape
    X_train = np.reshape(X_train, (-1, num_features))
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = np.reshape(X_train_scaled, (num_instances, num_time_steps, num_features))

    num_instances, num_time_steps, num_features = X_dev.shape
    X_dev = np.reshape(X_dev, (-1, num_features))
    X_dev_scaled = scaler.transform(X_dev)
    X_dev_scaled = np.reshape(X_dev_scaled, (num_instances, num_time_steps, num_features))

    num_instances, num_time_steps, num_features = X_test.shape
    X_test = np.reshape(X_test, (-1, num_features))
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.reshape(X_test_scaled, (num_instances, num_time_steps, num_features))
    
    for i in range(len(train_dataset)):
        train_dataset[i]['X'] = X_train_scaled[i]

    for i in range(len(dev_dataset)):
        dev_dataset[i]['X'] = X_dev_scaled[i]

    for i in range(len(test_dataset)):
        test_dataset[i]['X'] = X_test_scaled[i]

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory
    
    my_collator = MyCollator(batch_size=batch_size, seq_len=seq_len)

    train_dataloader = DataLoader(train_dataset, sampler=SequentialSampler(train_dataset), batch_size=batch_size*seq_len, num_workers=1, collate_fn=my_collator)
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size*seq_len, num_workers=1, collate_fn=my_collator)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size*seq_len, num_workers=1, collate_fn=MyCollator)

    return train_dataloader, dev_dataloader, test_dataloader

def main():
    #Parser
    parser = argparse.ArgumentParser(description='Fit an LSTM to the data, to predict the tweet counts from the embbedings.')

    parser.add_argument('--full_dataset_path', default=r"C:/Users/Filipa/Desktop/Predtweet/bitcoin_data/datasets/dt/", help="OS path to the folder where the embeddings are located.")
    parser.add_argument('--discretization_unit', default=1, help="The discretization unit is the number of hours to discretize the time series data. E.g.: If the user choses 3, then one sample point will cointain 3 hours of data.")
    parser.add_argument('--window_size', type = int, default=0, help='The window length defines how many units of time to look behind when calculating the features of a given timestamp.')
    parser.add_argument('--seq_len', type = int, default=10, help='Input dimension (number of timestamps).')
    parser.add_argument('--batch_size', type = int, default=3, help='How many batches of sequence length inputs per iteration.')
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.") 
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for Adam.") #5e-5
    #parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    #parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    #parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Warmup is the proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%")
    parser.add_argument("--model_name_or_path", default=r'/mnt/hdd_disk2/frente/finetuning_outputs/checkpoint-1', type=str, help="Path to folder containing saved checkpoints, schedulers, models, etc.")
    parser.add_argument("--output_dir", default='lstm/fit_results/', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Total number of training epochs to perform." )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_steps", type=int, default=300, help="Log every X updates steps.")
    parser.add_argument("--evaluate_during_training", action="store_false", help="Run evaluation during training at each logging step.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--test_acc", action="store_false", help="Run evaluation and store accuracy on test set.")
    
    args = parser.parse_args()
    print(args)

    path = args.full_dataset_path
    discretization_unit = args.discretization_unit
    window_size = args.window_size
    seq_len = args.seq_len
    batch_size = args.batch_size

    json_file_path = path+str(discretization_unit)+'.0/new_dataset.txt' 

    #load the dataset: timestamps and input ids (which correspond to the tweets already tokenized using BertTokenizerFast)
    #each chunk is read as a different dataset, and in the end all datasets are concatenated. A sequential sampler is defined.
    train_dataloader, dev_dataloader, test_dataloader = load_input(path = json_file_path, window_size=window_size, discretization_unit=discretization_unit, seq_len=seq_len, batch_size=batch_size)

    train_batch_size = 100
    val_batch_size = 100

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.isfile(args.output_dir+"/test_dataloader.pth"):
        torch.save(test_dataloader, args.output_dir+'/test_dataloader.pth')


    num_train_examples = len(train_dataloader)

    #Number of times the gradients are accumulated before a backward/update pass
    gradient_accumulation_steps = args.gradient_accumulation_steps

    # Number of training epochs, for finetuning in a specific NLP task, the authors recommend 2-4 epochs only
    epochs = args.num_train_epochs

    # input has dimension (samples, time steps, features)
    # create and fit the LSTM network
    EMBEDDING_DIM = 768 #number of features in data points
    HIDDEN_DIM = 64 #hidden dimension of the LSTM: number of nodes

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
    encoder = Encoder(EMBEDDING_DIM, HIDDEN_DIM, device)
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
   
        # Train the data for one epoch
        for step, batch in enumerate(epoch_iterator):  
    
            # Set our model to training mode (as opposed to evaluation mode)
            encoder = encoder.train()
            decoder = decoder.train() 

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            trainX_sample, trainY_sample = batch

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

            # Run our forward pass
            #scores = model(trainX_sample)

            # Zero gradients of both optimizers (by default they accumulate)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Reset hidden state of encoder for current batch
            # STATELESS LSTM
            if step==0:
                encoder.hidden = encoder.init_hidden(trainX_sample.shape[1])

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
                train_preds_seq.append(preds)
        
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

                if args.logging_steps>0 and global_step%args.logging_steps==0:
                    print("Train loss : {}".format(tr_loss/nb_tr_steps))
                    logs={}
                    
                    # DUVIDA: Pedir a zita para explicar isto
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    logs["loss"] = str(loss_scalar)
                    logging_loss = tr_loss

                    if args.evaluate_during_training: 
                        if global_step == args.logging_steps: #first evaluation: store validation observation sequence, do not need to overwrite it because it is fixed
                            results, val_obs_seq, val_preds_seq = evaluate(args, encoder, decoder, dev_dataloader, criterion, device, global_step, epoch, prefix = str(n_eval), store=True)
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
                        
                        
                        #Store 
                        val_loss_set.append((results["mse"], global_step, epoch)) 


                    print(json.dumps({**logs, **{"step": str(global_step)}}))

                if args.save_steps>0 and (global_step%args.save_steps==0 or save_best):
                    # Save model checkpoint
                    if save_best:
                        output_dir = os.path.join(args.output_dir, "best_model")
                        save_best = False
                    else:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    #model_to_save = (
                    #    encoder.module if hasattr(encoder, "module") else encoder
                    #)  # Take care of distributed/parallel training
                    #model_to_save.save(output_dir)
                    #model_to_save = (
                    #    decoder.module if hasattr(decoder, "module") else decoder
                    #)  # Take care of distributed/parallel training
                    #model_to_save.save(output_dir)
                    torch.save(encoder.state_dict(), output_dir+"/encoder.pth")
                    torch.save(decoder.state_dict(), output_dir+"/decoder.pth")

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
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
    plt.title("Training loss com batch size "+str(train_batch_size))
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    #plt.show()

    plt.savefig(os.path.join(args.output_dir)+'training_loss.png', bbox_inches='tight')

    # Plot validation loss (mse)
    plt.figure(figsize=(15,8))
    plt.title("Validation loss with batch size "+str(val_batch_size))
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
            best_encoder = Encoder(EMBEDDING_DIM, HIDDEN_DIM, device)
            best_decoder = Decoder(HIDDEN_DIM, device)

            #Put models in gpu
            best_encoder.cuda()
            best_decoder.cuda()
         
            
            #Load encoder and decoder states
            best_encoder.load_state_dict(torch.load(best_model_dir+"encoder.pth"))  
            best_decoder.load_state_dict(torch.load(best_model_dir+"decoder.pth"))
             
            results, test_obs_seq, test_preds_seq = evaluate(args, best_encoder, best_decoder, test_dataloader, criterion, device, global_step, epoch, prefix = 'Test', store=True)          

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