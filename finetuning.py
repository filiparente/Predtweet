import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import argparse
import numpy as np
from pathlib import Path
import json
from transformers import BertForSequenceClassification
from run_glue import basic_train
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from pytorch_pretrained_bert import BertAdam
from sentence_transformers import SentenceTransformer
from merge_json_files import load_inputids
from datetime import timedelta
import pandas as pd
from transformers import AdamW,  get_linear_schedule_with_warmup
import os
import logging

logger = logging.getLogger(__name__)

#current path
cpath = Path.cwd()

'''This function receives the time difference vector and calculates the weight for each timestamp,
depending on the temporal distance to the most recent timestamp. It uses an exponential function.
The user can specify the constants K and tau of the exponential, that are inputs of the function.'''
def weights(k,tau,timedif):  
    length = len(timedif)
    
    #Creates the weight vector with all zeros, the length is the same as the time difference vector
    #(as many weights as there are timestamps)
    weight_vector = np.zeros(length)
    
    #For each timestamp, calculate the weight associated to it based on the following exponential:
    #K*e^(-Td/tau) where Td is the temporal distance to the most recent event, in seconds
    for i in range(0,length,1):
        weight_vector[i] = k*(np.exp((-timedif[i])/tau))
    
    #Normalize the weights so that the weight vector has unit norm
    if sum(weight_vector) != 1:
        weight_vector = weight_vector/sum(weight_vector)
        
    return weight_vector

'''This function receives a vector of timestamps and returns a vector of the time difference 
(in seconds, the maximum is 10 seconds because is the maximum window duration)
between each timestamp and the most recent timestamp in the vector.
For example: [14h55, 14h53, 14h52, 14h47] will return [0, 2, 3, 8]'''
def timedifference(timestamps):
     #Creates the time difference vector
    timediff_vec = []
    
    #Obtains the timestamp of the most recent event (always in the first index)
    t1 = timestamps[0]
    
    i = 0
    
    #Cycle the timestamps vector
    while i < len(timestamps):
        
        #Obtain the next timestamp
        timei = timestamps[i]
        
        #Calculate the (positive) difference to the most recent timestamp (t1)
        dif = t1-timei
        
        #Convert it to seconds
        dif = dif.seconds
        
        #Append the difference in the time difference vector
        timediff_vec.append(dif)
        
        i=i+1
        
    return timediff_vec


def store(dataset, window_n, prev_date, next_date, delta, avg_emb, counts, store_embs):
    dataset['input_ids'].append({
        'id': window_n,
        'start_date': prev_date,
        'avg_emb': avg_emb,
        'count': counts,
    })
    window_n += 1

    store_embs = np.array([])
    counts = 0

    prev_date = next_date
    next_date = prev_date+delta

    return dataset, window_n, store_embs, counts, prev_date, next_date

def discretize_batch(batch, timestamps_, n_batch, delta, dataset, window_n, prev_date, next_date, store_embs, counts):   
    timestamps = pd.to_datetime(timestamps_)

    if n_batch == 1:
        prev_date = timestamps[0]
        next_date = prev_date+delta 
        store_embs = np.array([])
    
    end_date = timestamps_[-1]
    
    while(1):  
        mask = np.logical_and(timestamps>=prev_date, timestamps<next_date)

        if sum(mask)==0:
            dataset, window_n, store_embs, counts, prev_date, next_date = store(dataset, window_n, prev_date, next_date, delta, store_embs, counts, store_embs)
            continue

        counts += sum(mask)

        try:
            aux = batch['input_ids'][mask]
        except:
            print("error")

        if store_embs.size:
            try:
                avg_emb = torch.cat((store_embs, aux), 0)#np.concatenate(store_embs, aux)
            except:
                print("error")
        else:
            avg_emb = aux
        
        #if the last index is the date at the end of the batch, we need to open the next batch in order to
        #check if there are more input ids to store in the corresponding window
        if batch['timestamp'][mask][-1] == end_date:
            store_embs = avg_emb
            break

        dataset, window_n, store_embs, counts, prev_date, next_date = store(dataset, window_n, prev_date, next_date, delta, avg_emb, counts, store_embs)

    return dataset, window_n, prev_date, next_date, store_embs, counts
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    #pred_flat = np.argmax(preds, axis=1).flatten()
    #labels_flat = labels.flatten()
    #return np.sum(pred_flat == labels_flat) / len(labels_flat)
    return np.sum((preds - labels) ** 2)

def train_test_split(features, labels, percentages, window_size): 

    length_dataset = len(features) - 2*window_size #transition points between train/dev/test that are not valid because they contain information from two of them at the same time

    train_length = round(percentages[0]*length_dataset)
    dev_length = round(percentages[1]*length_dataset)
    test_length = round(percentages[2]*length_dataset)

    train_inputs = features[:train_length]
    train_labels = labels[:train_length]
    
    prev = train_length
    validation_inputs = features[prev+window_size:prev+window_size+dev_length]
    validation_labels = labels[prev+window_size:prev+window_size+dev_length]

    prev = train_length+window_size+dev_length
    test_inputs = features[prev+window_size:prev+window_size+test_length]
    test_labels = labels[prev+window_size:prev+window_size+test_length]

    return train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels
    
def main():
    parser = argparse.ArgumentParser(description='Finetune Bert for a regression task, to predict the tweet counts from the embbedings.')
    
    parser.add_argument('--dataset_path', default='bitcoin_data', help="OS path to the folder where the input ids are located.")
    parser.add_argument('--discretization_unit', default=1, help="The discretization unit is the number of hours to discretize the time series data. E.g.: If the user choses 3, then one sample point will cointain 3 hours of data.")
    parser.add_argument('--window_size', default=3, help="Number of time windows to look behind. E.g.: If the user choses 3, when to provide the features for the current window, we average the embbedings of the tweets of the 3 previous windows.")
    parser.add_argument("--save_steps", type=int, default=300, help="Save checkpoint every X updates steps.") #MUDAR ISTO PARA 300!!!!
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Warmup is the proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%")
    parser.add_argument("--model_name_or_path", default=r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\finetuning_outputs\checkpoint-10', type=str, help="Path to folder containing saved checkpoints, schedulers, models, etc.")
    parser.add_argument("--output_dir", default='finetuning_outputs', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    
    args = parser.parse_args()
    print(args) 

    dataset_path = cpath.joinpath(args.dataset_path)

    discretization_unit = args.discretization_unit
    delta = timedelta(hours=discretization_unit)
    window_size = args.window_size

    #Calculates the timedifference
    timedif = [i for i in range(window_size)]

    #Calculate the weights using K = 0.5 (giving 50% of importance to the most recent timestamp)
    #and tau = 6.25s so that when the temporal difference is 10s, the importance is +- 10.1%
    wi = weights(0.5, 2, timedif)

    # Approximation of batch_size
    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32, which represent 32 sample points (features, embbedings) after preprocessing is done
    batch_size_ = 16 #32 
    approx = 60 #~ number of tweets in one hour
    batch_size = round(batch_size_+window_size*discretization_unit*approx-1)

    #load the dataset: timestamps and input ids (which correspond to the tweets already tokenized using BertTokenizerFast)
    #each chunk is read as a different dataset, and in the end all datasets are concatenated. A sequential sampler is defined.
    train_dataloader, dev_dataloader, test_dataloader = load_inputids(path = dataset_path, batch_size=batch_size)

    num_train_examples = int(1653*0.8)

    #batch_size = 5 #approximate

    #Number of times the gradients are accumulated before a backward/update pass
    gradient_accumulation_steps = 1

    # Number of training epochs, for finetuning in a specific NLP task, the authors recommend 2-4 epochs only
    epochs = 4

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path): #Path to pre-trained model
        # set global_step to global_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (num_train_examples // gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (num_train_examples // gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

    model.cuda()

    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    num_train_optimization_steps = int(num_train_examples/batch_size/gradient_accumulation_steps)*epochs

    warmup_steps = args.warmup_proportion*num_train_optimization_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta", "LayerNorm.weight"] #gamma and beta were not in the run_glue.py
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0} 
    ]
    # This variable contains all of the hyperparemeter information our training loop needs
    #optimizer = BertAdam(optimizer_grouped_parameters,
    #                    lr=2e-5, #The initial learning rate for Adam, default is 5e-5
    #                    warmup=.1, #warmup is the proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%"
    #                    t_total=num_train_optimization_steps)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))


    # Store our loss and accuracy for plotting
    train_loss_set = []
    dev_acc_set = []

    global_step = 0

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
    
        # Training   
        # Set our model to training mode (as opposed to evaluation mode)
        model = model.train()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        window_n = 1
        counts = 0
        prev_date = None
        next_date = None
        store_embs = np.array([])

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):       
            #with torch.no_grad():   #   depois tirar isto!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #Get timestamps
            timestamps_ = batch['timestamp']
    
            dataset = {}
            dataset['disc_unit'] = discretization_unit
            dataset['window_size'] = window_size
            dataset['input_ids'] = []

            dataset, window_n, prev_date, next_date, store_embs, counts = discretize_batch(batch, timestamps_, step+1, \
            delta, dataset, window_n, prev_date, next_date, store_embs, counts) #step+1 because enumeration starts at 0 index

            #For each individual timestamp
            if window_size>len(dataset['input_ids']):
                print("ERROR. WINDOW_SIZE IS TOO BIG!")
            else:                                                 
                idx = window_size
                length = len(dataset['input_ids'])
                X = [np.array([]) for i in range(length-window_size)]
                y = np.zeros(length-window_size)
                nn = 0
                while idx < length:
                    start = dataset['input_ids'][idx]
            
                    X[nn] = []
                    for i in range(1,1+window_size):

                        X[nn].append({'weight':wi[i-1], 'input_ids': torch.stack([vec.type(torch.LongTensor) for vec in dataset['input_ids'][idx-i]['avg_emb']]).to(device)}) 
            
                    y[nn] = int(start['count'])
                    nn+=1
                    idx += 1

            del dataset
            print("Number of examples in training batch n" + str(step)+" : " + str(len(X)))

            # Clear out the gradients (by default they accumulate)
            #optimizer.zero_grad() #DUVIDA: ISTO E PARA TIRAR?

            # Forward pass
            if len(X)>=1: #the batch must contain, at least, one example, otherwise don't do forward
                loss, logits = model(input_ids = X, labels=torch.tensor(y).to(device), weights=wi, window_size=window_size)
                train_loss_set.append(loss.item())    

                a = list(model.parameters())[199].clone()
                a2 = list(model.parameters())[200].clone()

                #if len(X)>=1:  #the batch must contain, at least, one example, otherwise don't do backward and don't update anything
                # Backward pass
                loss.backward()

                # Update parameters and take a step using the computed gradient
                optimizer.step()
                scheduler.step() #update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps>0 and global_step%args.save_steps==0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)


                b = list(model.parameters())[199].clone()
                b2 = list(model.parameters())[200].clone()
                
                # PARA CONFIRMAR SE OS PESOS ESTAVAM A SER ALTERADOS OU NAO: 199 e o peso W e 200 e o peso b (bias) da layer de linear de classificacao/regressao: WX+b
                if step%1==0:
                    print("Check if the classifier layer weights are being updated:")
                    print("Weight W: "+str(not torch.equal(a.data, b.data)))  
                    print("Bias b: " + str(not torch.equal(a2.data, b2.data)))
                    
                
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += len(X) #b_input.size(0)
                nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
          
        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model = model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for step, batch in dev_dataloader:
            # Add batch to GPU
            #batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            #b_input, b_labels = batch
            #Get timestamps
            timestamps_ = batch['timestamp']
        
            dataset = {}
            dataset['disc_unit'] = discretization_unit
            dataset['window_size'] = window_size
            dataset['input_ids'] = []

            dataset, window_n, prev_date, next_date, store_embs, counts = discretize_batch(batch, timestamps_, step+1, \
            delta, dataset, window_n, prev_date, next_date, store_embs, counts) #step+1 because enumeration starts at 0 index

            #For each individual timestamp
            if window_size>len(dataset['input_ids']):
                print("ERROR. WINDOW_SIZE IS TOO BIG!")
            else:                                                 
                idx = window_size
                length = len(dataset['input_ids'])
                X = [np.array([]) for i in range(length-window_size)]
                y = np.zeros(length-window_size)
                nn = 0
                while idx < length:
                    start = dataset['input_ids'][idx]
                
                    X[nn] = []
                    for i in range(1,1+window_size):

                        X[nn].append({'weight':wi[i-1], 'input_ids': torch.stack([vec.type(torch.LongTensor) for vec in dataset['input_ids'][idx-i]['avg_emb']]).to(device)}) 
                
                    y[nn] = int(start['count'])
                    nn+=1
                    idx += 1

            del dataset
            print("Number of examples in validation batch n" + str(step)+" : " + str(len(X)))
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            if len(X)>=1:
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    logits = model(input_ids = X, weights=wi, window_size=window_size)
                    
                # Move logits and labels to CPU
                logits = logits[0].detach().cpu().numpy()
                #print(logits)
                label_ids = y

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                dev_acc_set.append(tmp_eval_accuracy)  

                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))


    plt.figure(figsize=(15,8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    plt.show()

    plt.figure(figsize=(15,8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(dev_acc_set)
    plt.show()


def load_dataset(dataset_path):
    with open(dataset_path,encoding='utf-8', errors='ignore', mode='r') as j:
        data = json.loads(j.read())

        features = [el['X'] for el in data['embeddings']]
        labels = [el['y'] for el in data['embeddings']]

        window_size = data['window_size']

        return features, labels, window_size

if __name__ == '__main__':
    main()
    
