#    1) Reportar 6 métricas: 
#    MSE normal, 
#    RMSE normal, 
#    RMSE normalizado pela média das observações y, 
#    RMSE normalizado por ymax-ymin,
#    RMSE normalizado pela std(y),
#    RMSE normalizado pela diferença dos quantiles 0.75 e 0.25 de y,
#    FFT
#
#    2) guardar numa estrutura BERT_runx_prediction_report.mat
# TODO: THIS IS ADDAPTED FOR NO EXTERNAL FEATURES!, to use with external features, adapt the code first

import random
import torch
import numpy as np
import os
import glob
import json
import argparse
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import pandas as pd
import scipy
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.io import savemat

class LSTM(nn.Module):
    #input size: Corresponds to the number of features in the input. Though our sequence length is 12, for each month we have only 1 value i.e. total number of passengers, therefore the input size will be 1.
    #hidden layer size: Specifies the number of hidden layers along with the number of neurons in each layer. We will have one layer of 100 neurons.
    #output size: The number of items in the output, since we want to predict the number of passengers for 1 month in the future, the output size will be 1.´

    def __init__(self, device, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    def init_hidden(self, batch_size):
        #return (nn.Parameter(torch.randn(self.num_layers,batch_size, self.hidden_dim).type(torch.FloatTensor).to(self.device), requires_grad=True), nn.Parameter(torch.randn(self.num_layers, batch_size, self.hidden_dim).type(torch.FloatTensor).to(self.device),requires_grad=True))

        return (torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(self.device), #hidden state
                torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(self.device)) #cell state
       

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

# convert an array of values into a dataset matrix
def create_dataset(X,y, batch_size, seq_len):
    #dataX, dataY = [], []
    idx = 0
    #auxX = np.zeros((batch_size, seq_len, 768))
    #auxY = np.zeros((batch_size, seq_len))
    #dataset = dict()
    #dataset['X'] = np.array([])
    #dataset['y'] = np.array([])
    dataset = []

    if len(X)>=batch_size*seq_len:
        
        while(1):
            auxX = np.zeros((batch_size, seq_len, 768))
            auxY = np.zeros((batch_size, seq_len))
            for i in range(batch_size):
                auxX[i,:,:] = X[idx+i*seq_len:seq_len+idx+i*seq_len]
                auxY[i,:] = y[idx+i*seq_len:seq_len+idx+i*seq_len]
            
            dataset.append({'X':np.array(auxX), 'y':np.array(auxY)})

            #dataX.append(auxX)
            #dataY.append(auxY)
            if seq_len+idx+i*seq_len == len(X):
                break
            idx=idx+1
            del auxX
            del auxY
            

        #dataset['X'] = np.array(dataX)
        #dataset['y'] = np.array(dataY)
    
    return dataset

def create_dataset2(sequence, lag=1):
    #(seq_len, batch_size, input_size)=(1,3786,1)
    X, y = sequence[0:-lag], sequence[lag:]
    
    return np.reshape(np.array(X), (1,len(sequence)-1,1)), np.reshape(np.array(y), (1,len(sequence)-1))

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

def fft_(signal1, signal2, plot_):
    time = range(1,len(signal1))                         # Time Vector                               # Signal data in Time-Domain
    N = len(signal1)                                # Number Of Samples
    Ts = np.mean(np.diff(time))                              # Sampling Interval
    Fs = 1/Ts                                          # Sampling Frequency
    Fn = Fs/2                                          # Nyquist Frequency
    Fv = np.linspace(0, 1, int(float(N/2))+1)*Fn                 # Frequency Vector (For ‘plot’ Call)
    Iv = range(1,len(Fv))                                # Index Vector (Matches ‘Fv’)
    
    FT_Signal1 = scipy.fft(signal1)/N                        # Normalized Fourier Transform Of Data
    FT_Signal2 = scipy.fft(signal2)/N                        # Normalized Fourier Transform Of Data
    
    if plot_:
        plt.figure(figsize=(15,8))
        plt.plot(Fv, abs(FT_Signal1(Iv))*2)
        plt.plot(Fv, abs(FT_Signal2(Iv))*2)
        plt.show()
   
    
    #Mean squared errors
    #Phase
    tmp = pow((np.transpose(np.angle(FT_Signal1))-np.angle(FT_Signal2)),2)
    srmse_phase = np.sqrt(sum(tmp[:])/N)/np.std(FT_Signal1)
    
    #Phase
    tmp = pow((np.transpose(np.real(FT_Signal1))-np.real(FT_Signal2)),2)
    srmse_ampl = np.sqrt(sum(tmp[:])/N)/np.std(FT_Signal1)
    
def main():
    #Parser
    parser = argparse.ArgumentParser(description='Normalize LSTM errors.')
    parser.add_argument('--model_path', default=r"C:/Users/Filipa/Desktop/Predtweet/lstm/arquitetura2/full_data/wt_features/", help="OS path to the folder where the embeddings are located.")
    parser.add_argument('--full_dataset_path', default=r"C:/Users/Filipa/Desktop/Predtweet/bitcoin_data/datasets/server/1.0/", help="OS path to the folder where the embeddings are located.")
    parser.add_argument('--seq_len', type = int, default=50, help='Input dimension (number of timestamps).')
    parser.add_argument('--batch_size', type = int, default=1, help='How many batches of sequence length inputs per iteration.')
    parser.add_argument("--use_features", action="store_true", help="If we want to consider the textual features (from BERT/TFIDF) or only the counts.")
    parser.add_argument("--output_dir", default=r"C:/Users/Filipa/Desktop/Predtweet/lstm/arquitetura2/full_data/")
    args = parser.parse_args()
    print(args)
    
    model_path = args.model_path    
    path = args.full_dataset_path
    train_dev_test_split = True
    normalize = True
    batch_size = args.batch_size
    seq_len = args.seq_len
    percentages = [0.05, 0.05, 0.05]
    
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

    
    run = input("Save prediction report for which run (accepted input: integer number 1,2,...) ?")

    dts = [1] #[1,3,4,6] #[1,3,4,6,12,24,48]
    dws = [0]#,1,3,5,7] 
    
    result = {}

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

    #plt.plot(obs_seq)
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
                

        #lengths = np.cumsum(lengths)
        #lengths = [int(l) for l in lengths]

        #train_obs_seq = all_obs_seq[:lengths[0]]
        #train_X = all_X[:, :lengths[0]]
        #dev_obs_seq = all_obs_seq[lengths[0]+window_size:lengths[1]]
        #dev_X = all_X[:, lengths[0]+window_size:lengths[1]]
        #test_obs_seq = all_obs_seq[lengths[1]+window_size:]
        #test_X = all_X[:, lengths[1]+window_size:]

        lengths = list(np.insert(np.cumsum(lengths)+len_dataset-sum(lengths), 0, len_dataset-sum(lengths)))
        lengths = [int(l) for l in lengths]

        train_obs_seq = all_obs_seq[lengths[0]:lengths[1]]
        train_X = all_X[:, lengths[0]:lengths[1]]
        dev_obs_seq = all_obs_seq[lengths[1]+window_size:lengths[2]]
        dev_X = all_X[:, lengths[1]+window_size:lengths[2]]
        test_obs_seq = all_obs_seq[lengths[2]+window_size:]
        test_X = all_X[:, lengths[2]+window_size:]
                

        #lengths = np.cumsum(lengths)
        #lengths = [int(l) for l in lengths]

        #train_obs_seq = all_obs_seq[:lengths[0]]
        #train_X = all_X[:, :lengths[0]]
        #dev_obs_seq = all_obs_seq[lengths[0]+window_size:lengths[1]]
        #dev_X = all_X[:, lengths[0]+window_size:lengths[1]]
        #test_obs_seq = all_obs_seq[lengths[1]+window_size:]
        #test_X = all_X[:, lengths[1]+window_size:]
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
            #maxV = max(train_X(feature,:)) 
            #minV = min(train_X(feature,:)) 
            #train_X(feature,:)   = (train_X(feature,:) - minV) / (maxV - minV) 
            
            dev_X[feature,:]=(dev_X[feature,:]-train_mean[feature])/train_std[feature]
            test_X[feature,:]=(test_X[feature,:]-train_mean[feature])/train_std[feature]
        

    train_dataset = MyDataset(train_obs_seq, train_X, window_size, seq_len)
    dev_dataset = MyDataset(dev_obs_seq, dev_X, window_size, seq_len)
    test_dataset = MyDataset(test_obs_seq, test_X, window_size, seq_len)

    if args.use_features:
        train_dataset = create_dataset(train_X, train_obs_seq, batch_size, seq_len)
        dev_dataset = create_dataset(dev_X, dev_obs_seq, batch_size, seq_len)
        test_dataset = create_dataset(test_X, test_obs_seq, batch_size, seq_len)

        assert len(train_dataset)>0, "Batch size or sequence length too big for training set!"
        assert len(dev_dataset)>0, "Batch size or sequence length too big for validation set!"
        assert len(test_dataset)>0, "Batch size or sequence length too big for test set!"

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
        # with an iterator the entire dataset does not need to be loaded into memory

        print("Number of points in converted train dataset = " + str(len(train_dataset))+ " with sliding batches with batch_size "+ str(batch_size) +" and sequence length "+str(seq_len))
        print("Number of points in converted dev dataset = " + str(len(dev_dataset))+ " with sliding batches with batch_size "+ str(batch_size) +" and sequence length "+str(seq_len))
        print("Number of points in converted test dataset = " + str(len(test_dataset))+ " with sliding batches with batch_size "+ str(batch_size) +" and sequence length "+str(seq_len))

    else:
        #normalize data: min/max scaling (-1 and 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = scaler.fit_transform(train_obs_seq.reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

        #sequence/labeling
        #input sequence length for training is 24. (1h data, 24h memory)
        train_window = 24

        train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


        train_dataset = create_dataset2(train_obs_seq)
        dev_dataset = create_dataset2(np.concatenate(([train_obs_seq[-1]],dev_obs_seq)))
        test_dataset = create_dataset2(np.concatenate(([dev_obs_seq[-1]],test_obs_seq)))

        assert len(train_dataset[0])>0, "Batch size or sequence length too big for training set!"
        assert len(dev_dataset[0])>0, "Batch size or sequence length too big for validation set!"
        assert len(test_dataset[0])>0, "Batch size or sequence length too big for test set!"

        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
        # with an iterator the entire dataset does not need to be loaded into memory

        print("Number of points in converted train dataset = " + str(len(train_dataset[0]))+ " with sliding batches with batch_size "+ str(batch_size) +" and sequence length "+str(seq_len))
        print("Number of points in converted dev dataset = " + str(len(dev_dataset[0]))+ " with sliding batches with batch_size "+ str(batch_size) +" and sequence length "+str(seq_len))
        print("Number of points in converted test dataset = " + str(len(test_dataset[0]))+ " with sliding batches with batch_size "+ str(batch_size) +" and sequence length "+str(seq_len))
        
        mse_train = []
        mae_train = []
        mse_dev = []
        mae_dev = []
        mse_test = []
        mae_test = []

        train_window = 24
        n_montecarlos = 10
        dw=0

        #Load results
        for i in range(len(dts)):
            dt = dts[i]

            #for j in range(len(dws)):
            #    dw = dws[j]
                
            for montecarlo in range(n_montecarlos):    
                #Results
                #Train predictions
                #train_preds_seq = np.concatenate(torch.load(model_path + str(dt) + '.' + str(dw) +'/run1_results/best_model/train_preds_seq.pt')).ravel().tolist()
                train_preds_seq = np.concatenate(torch.load(model_path + str(dt) + '.' + str(dw) +'/5_5_5/tensors/train_preds_seq_mc' + str(montecarlo) + '.pt')).ravel().tolist()

                #Validation predictions
                #best_val_preds_seq = np.concatenate(torch.load(model_path + str(dt) + '.' + str(dw) +'/run1_results/best_model/best_val_preds_seq.pt')).ravel().tolist()
                best_val_preds_seq = np.concatenate(torch.load(model_path + str(dt) + '.' + str(dw) +'/5_5_5/tensors/best_val_preds_seq_mc' + str(montecarlo) + '.pt')).ravel().tolist()

                #Test predictions
                #test_preds_seq = np.concatenate(torch.load(model_path + str(dt) + '.' + str(dw) +'/run1_results/best_model/test_preds_seq_mc.pt')).ravel().tolist()
                test_preds_seq = np.concatenate(torch.load(model_path + str(dt) + '.' + str(dw) +'/5_5_5/tensors/test_preds_seq_mc' + str(montecarlo) + '.pt')).ravel().tolist()

                mae_test.append(np.mean((abs(test_preds_seq-test_obs_seq))/test_obs_seq)*100)
                #mse
                #mse_train.append(mean_squared_error(train_obs_seq[train_window:], train_preds_seq))
                #mse_dev.append(mean_squared_error(dev_obs_seq[dw:], best_val_preds_seq))
                #mse_test.append(mean_squared_error(test_obs_seq[dw:], test_preds_seq))

                #mae
                #mae_train.append(mean_absolute_error(train_obs_seq[train_window:], train_preds_seq))
                #mae_dev.append(mean_absolute_error(dev_obs_seq[dw:], best_val_preds_seq))
                #mae_test.append(mean_absolute_error(test_obs_seq[dw:], test_preds_seq))

        #MSE
        print(np.mean(mae_test))
        mse_mean_train = np.mean(mse_train)
        mse_std_train = np.std(mse_train)
        mse_mean_dev = np.mean(mse_dev)
        mse_std_dev = np.std(mse_dev)
        mse_mean_test = np.mean(mse_test)
        mse_std_test = np.std(mse_test)


        #rmse
        rmse_mean_train = np.sqrt(mse_mean_train)
        rmse_std_train = np.sqrt(mse_std_train)
        rmse_mean_dev = np.sqrt(mse_mean_dev)
        rmse_std_dev = np.sqrt(mse_std_dev)
        rmse_mean_test = np.sqrt(mse_mean_test)
        rmse_std_test = np.sqrt(mse_std_test)

        #normalize by mean y
        mnrmse_mean_train = rmse_mean_train/np.mean(train_obs_seq)
        mnrmse_std_train = rmse_std_train/np.mean(train_obs_seq)
        mnrmse_mean_dev = rmse_mean_dev/np.mean(dev_obs_seq)
        mnrmse_std_dev = rmse_std_dev/np.mean(dev_obs_seq)
        mnrmse_mean_test = rmse_mean_test/np.mean(test_obs_seq)
        mnrmse_std_test = rmse_std_test/np.mean(test_obs_seq)

        #normalize by max(y)-min(y)
        mmnrmse_mean_train = rmse_mean_train/(max(train_obs_seq)-min(train_obs_seq))
        mmnrmse_std_train = rmse_std_train/(max(train_obs_seq)-min(train_obs_seq))
        mmnrmse_mean_dev = rmse_mean_dev/(max(dev_obs_seq)-min(dev_obs_seq))
        mmnrmse_std_dev = rmse_std_dev/(max(dev_obs_seq)-min(dev_obs_seq))
        mmnrmse_mean_test = rmse_mean_test/(max(test_obs_seq)-min(test_obs_seq))
        mmnrmse_std_test = rmse_std_test/(max(test_obs_seq)-min(test_obs_seq))
        
        #normalize by std(y)
        snrmse_mean_train = rmse_mean_train/np.std(train_obs_seq)
        snrmse_std_train = rmse_std_train/np.std(train_obs_seq)
        snrmse_mean_dev = rmse_mean_dev/np.std(dev_obs_seq)
        snrmse_std_dev = rmse_std_dev/np.std(dev_obs_seq)
        snrmse_mean_test = rmse_mean_test/np.std(test_obs_seq)
        snrmse_std_test = rmse_std_test/np.std(test_obs_seq)
        
        #normalize by quantile(y,0.75)-quantile(y,0.25)
        qnrmse_mean_train = rmse_mean_train/(np.quantile(train_obs_seq,0.75)-np.quantile(train_obs_seq,0.25))
        qnrmse_std_train = rmse_std_train/(np.quantile(train_obs_seq,0.75)-np.quantile(train_obs_seq,0.25))
        qnrmse_mean_dev = rmse_mean_dev/(np.quantile(dev_obs_seq,0.75)-np.quantile(dev_obs_seq,0.25))
        qnrmse_std_dev = rmse_std_dev/(np.quantile(dev_obs_seq,0.75)-np.quantile(dev_obs_seq,0.25))
        qnrmse_mean_test = rmse_mean_test/(np.quantile(test_obs_seq,0.75)-np.quantile(test_obs_seq,0.25))
        qnrmse_std_test = rmse_std_test/(np.quantile(test_obs_seq,0.75)-np.quantile(test_obs_seq,0.25))

        #MAE
        mae_mean_train = np.mean(mae_train)
        mae_std_train = np.std(mae_train)
        mae_mean_dev = np.mean(mae_dev)
        mae_std_dev = np.std(mae_dev)
        mae_mean_test = np.mean(mae_test)
        mae_std_test = np.std(mae_test)


        #rmse
        rmae_mean_train = np.sqrt(mae_mean_train)
        rmae_std_train = np.sqrt(mae_std_train)
        rmae_mean_dev = np.sqrt(mae_mean_dev)
        rmae_std_dev = np.sqrt(mae_std_dev)
        rmae_mean_test = np.sqrt(mae_mean_test)
        rmae_std_test = np.sqrt(mae_std_test)

        #normalize by mean y
        mnrmae_mean_train = rmae_mean_train/np.mean(train_obs_seq)
        mnrmae_std_train = rmae_std_train/np.mean(train_obs_seq)
        mnrmae_mean_dev = rmae_mean_dev/np.mean(dev_obs_seq)
        mnrmae_std_dev = rmae_std_dev/np.mean(dev_obs_seq)
        mnrmae_mean_test = rmae_mean_test/np.mean(test_obs_seq)
        mnrmae_std_test = rmae_std_test/np.mean(test_obs_seq)

        #normalize by max(y)-min(y)
        mmnrmae_mean_train = rmae_mean_train/(max(train_obs_seq)-min(train_obs_seq))
        mmnrmae_std_train = rmae_std_train/(max(train_obs_seq)-min(train_obs_seq))
        mmnrmae_mean_dev = rmae_mean_dev/(max(dev_obs_seq)-min(dev_obs_seq))
        mmnrmae_std_dev = rmae_std_dev/(max(dev_obs_seq)-min(dev_obs_seq))
        mmnrmae_mean_test = rmae_mean_test/(max(test_obs_seq)-min(test_obs_seq))
        mmnrmae_std_test = rmae_std_test/(max(test_obs_seq)-min(test_obs_seq))
        
        #normalize by std(y)
        snrmae_mean_train = rmae_mean_train/np.std(train_obs_seq)
        snrmae_std_train = rmae_std_train/np.std(train_obs_seq)
        snrmae_mean_dev = rmae_mean_dev/np.std(dev_obs_seq)
        snrmae_std_dev = rmae_std_dev/np.std(dev_obs_seq)
        snrmae_mean_test = rmae_mean_test/np.std(test_obs_seq)
        snrmae_std_test = rmae_std_test/np.std(test_obs_seq)
        
        #normalize by quantile(y,0.75)-quantile(y,0.25)
        qnrmae_mean_train = rmae_mean_train/(np.quantile(train_obs_seq,0.75)-np.quantile(train_obs_seq,0.25))
        qnrmae_std_train = rmae_std_train/(np.quantile(train_obs_seq,0.75)-np.quantile(train_obs_seq,0.25))
        qnrmae_mean_dev = rmae_mean_dev/(np.quantile(dev_obs_seq,0.75)-np.quantile(dev_obs_seq,0.25))
        qnrmae_std_dev = rmae_std_dev/(np.quantile(dev_obs_seq,0.75)-np.quantile(dev_obs_seq,0.25))
        qnrmae_mean_test = rmae_mean_test/(np.quantile(test_obs_seq,0.75)-np.quantile(test_obs_seq,0.25))
        qnrmae_std_test = rmae_std_test/(np.quantile(test_obs_seq,0.75)-np.quantile(test_obs_seq,0.25))
        
        #FFT dev
        #[fft_srmse_phase_dev, fft_srmse_ampl_dev] = fft_(dev_obs_seq, best_val_preds_seq, False)
        
        #FFT test
        #[fft_srmse_phase_test, fft_srmse_ampl_test] = fft_(test_obs_seq, test_preds_seq, False)


        #result['dt'] = dt
        #result['dw'] = dw
        #result['fft_mse_phase_dev'] = fft_srmse_phase_dev
        #result['fft_mse_ampl_dev'] = fft_srmse_ampl_dev
        # result['fft_mse_phase_test'] = fft_srmse_phase_test
        # result['fft_mse_ampl_test'] = fft_srmse_ampl_test
        
        mse =  dict()
        mse['mean_train'] = mse_mean_train 
        mse['std_train'] = mse_std_train 
        mse['mean_dev'] = mse_mean_dev 
        mse['std_dev'] = mse_std_dev 
        mse['mean_test'] = mse_mean_test 
        mse['std_test'] = mse_std_test 


        result['mse'] = mse 

        rmse =  dict()
        rmse['mean_train'] = rmse_mean_train 
        rmse['std_train'] = rmse_std_train 
        rmse['mean_dev'] = rmse_mean_dev 
        rmse['std_dev'] = rmse_std_dev 
        rmse['mean_test'] = rmse_mean_test 
        rmse['std_test'] = rmse_std_test 

        result['rmse'] = rmse 

        mnrmse =  dict()
        mnrmse['mean_train'] = mnrmse_mean_train 
        mnrmse['std_train'] = mnrmse_std_train 
        mnrmse['mean_dev'] = mnrmse_mean_dev 
        mnrmse['std_dev'] = mnrmse_std_dev 
        mnrmse['mean_test'] = mnrmse_mean_test 
        mnrmse['std_test'] = mnrmse_std_test 

        result['mnrmse'] = mnrmse 

        mmnrmse =  dict()
        mmnrmse['mean_train'] = mmnrmse_mean_train 
        mmnrmse['std_train'] = mmnrmse_std_train 
        mmnrmse['mean_dev'] = mmnrmse_mean_dev 
        mmnrmse['std_dev'] = mmnrmse_std_dev 
        mmnrmse['mean_test'] = mmnrmse_mean_test 
        mmnrmse['std_test'] = mmnrmse_std_test 

        result['mmnrmse'] = mmnrmse 

        snrmse =  dict()
        snrmse['mean_train'] = snrmse_mean_train 
        snrmse['std_train'] = snrmse_std_train 
        snrmse['mean_dev'] = snrmse_mean_dev 
        snrmse['std_dev'] = snrmse_std_dev 
        snrmse['mean_test'] = snrmse_mean_test 
        snrmse['std_test'] = snrmse_std_test 

        result['snrmse'] = snrmse 

        qnrmse = dict()
        qnrmse['mean_train'] = qnrmse_mean_train 
        qnrmse['std_train'] = qnrmse_std_train 
        qnrmse['mean_dev'] = qnrmse_mean_dev
        qnrmse['std_dev'] = qnrmse_std_dev
        qnrmse['mean_test'] = qnrmse_mean_test
        qnrmse['std_test'] = qnrmse_std_test

        result['qnrmse'] = qnrmse

        #MAE
        mae =  dict()
        mae['mean_train'] = mae_mean_train 
        mae['std_train'] = mae_std_train 
        mae['mean_dev'] = mae_mean_dev 
        mae['std_dev'] = mae_std_dev 
        mae['mean_test'] = mae_mean_test 
        mae['std_test'] = mae_std_test 


        result['mae'] = mae 

        rmae =  dict()
        rmae['mean_train'] = rmae_mean_train 
        rmae['std_train'] = rmae_std_train 
        rmae['mean_dev'] = rmae_mean_dev 
        rmae['std_dev'] = rmae_std_dev 
        rmae['mean_test'] = rmae_mean_test 
        rmae['std_test'] = rmae_std_test 

        result['rmae'] = rmae 

        mnrmae =  dict()
        mnrmae['mean_train'] = mnrmae_mean_train 
        mnrmae['std_train'] = mnrmae_std_train 
        mnrmae['mean_dev'] = mnrmae_mean_dev 
        mnrmae['std_dev'] = mnrmae_std_dev 
        mnrmae['mean_test'] = mnrmae_mean_test 
        mnrmae['std_test'] = mnrmae_std_test 

        result['mnrmae'] = mnrmae 

        mmnrmae =  dict()
        mmnrmae['mean_train'] = mmnrmae_mean_train 
        mmnrmae['std_train'] = mmnrmae_std_train 
        mmnrmae['mean_dev'] = mmnrmae_mean_dev 
        mmnrmae['std_dev'] = mmnrmae_std_dev 
        mmnrmae['mean_test'] = mmnrmae_mean_test 
        mmnrmae['std_test'] = mmnrmae_std_test 

        result['mmnrmae'] = mmnrmae 

        snrmae =  dict()
        snrmae['mean_train'] = snrmae_mean_train 
        snrmae['std_train'] = snrmae_std_train 
        snrmae['mean_dev'] = snrmae_mean_dev 
        snrmae['std_dev'] = snrmae_std_dev 
        snrmae['mean_test'] = snrmae_mean_test 
        snrmae['std_test'] = snrmae_std_test 

        result['snrmae'] = snrmae 

        qnrmae = dict()
        qnrmae['mean_train'] = qnrmae_mean_train 
        qnrmae['std_train'] = qnrmae_std_train 
        qnrmae['mean_dev'] = qnrmae_mean_dev
        qnrmae['std_dev'] = qnrmae_std_dev
        qnrmae['mean_test'] = qnrmae_mean_test
        qnrmae['std_test'] = qnrmae_std_test

        result['qnrmae'] = qnrmae

        #Save BERT_runx_prediction_report.mat
        #save([path, 'BERT_run' num2str(run) '_prediction_report.mat'], 'out_results')

        savemat(args.output_dir + 'run' + str(run) + '_prediction_report_LSTM.mat', result, oned_as='row')

if __name__=='__main__':
    main()