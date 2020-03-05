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
import random
from sklearn.metrics import mean_squared_error

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

#current path
cpath = Path.cwd()

class TweetBatch():
    def __init__(self, discretization_unit, window_size):
        dataset = {}
        dataset['disc_unit'] = discretization_unit
        dataset['window_size'] = window_size
        dataset['input_ids'] = []

        self.delta = timedelta(hours=discretization_unit)
        self.dataset = dataset
        self.window_n = 1
        self.prev_date = None
        self.next_date = None
        self.store_embs = np.array([])
        self.counts = 0

    def discretize_batch(self, batch, step, n_batch):
        #Get timestamps
        timestamps_ = batch['timestamp']

        timestamps = pd.to_datetime(timestamps_)

        if n_batch == 1:
            self.prev_date = timestamps[0]
            self.next_date = self.prev_date+self.delta 
            self.store_embs = np.array([])
        
        end_date = timestamps_[-1]
        
        while(1):  
            mask = np.logical_and(timestamps>=self.prev_date, timestamps<self.next_date)

            if sum(mask)==0:
                #dataset, window_n, store_embs, counts, prev_date, next_date = store(dataset, window_n, prev_date, next_date, delta, store_embs, counts, store_embs)
                self.store(self.store_embs)
                continue

            self.counts += sum(mask)

            try:
                aux = batch['input_ids'][mask]
            except:
                print("error")

            if self.store_embs.size:
                try:
                    avg_emb = torch.cat((self.store_embs, aux), 0)#np.concatenate(store_embs, aux)
                except:
                    print("error")
            else:
                avg_emb = aux
            
            #if the last index is the date at the end of the batch, we need to open the next batch in order to
            #check if there are more input ids to store in the corresponding window
            if batch['timestamp'][mask][-1] == end_date:
                self.store_embs = avg_emb
                break

            #dataset, window_n, store_embs, counts, prev_date, next_date = store(dataset, window_n, prev_date, next_date, delta, avg_emb, counts, store_embs)
            self.store(avg_emb)

        #return dataset, window_n, prev_date, next_date, store_embs, counts
    
    def store(self, avg_emb):
        self.dataset['input_ids'].append({
            'id': self.window_n,
            'start_date': self.prev_date,
            'avg_emb': avg_emb,
            'count': self.counts,
        })
        self.window_n += 1

        self.store_embs = np.array([])
        self.counts = 0

        self.prev_date = self.next_date
        self.next_date = self.prev_date+self.delta

        #return dataset, window_n, store_embs, counts, prev_date, next_date

    def sliding_window(self, wi, device, step):
        #For each individual timestamp
        window_size = self.dataset['window_size']
        length = len(self.dataset['input_ids'])

        if window_size >= length:
            print("ERROR. WINDOW_SIZE IS TOO BIG! Loading next tweet batch...")
        else:                                                 
            idx = window_size
            X = [np.array([]) for i in range(length-window_size)]
            y = np.zeros(length-window_size)
            nn = 0
            while idx < length:
                start = self.dataset['input_ids'][idx]
        
                X[nn] = []
                for i in range(1,1+window_size):

                    X[nn].append({'weight':wi[i-1], 'input_ids': torch.stack([vec.type(torch.LongTensor) for vec in self.dataset['input_ids'][idx-i]['avg_emb']]).to(device)}) 
        
                y[nn] = int(start['count'])
                nn+=1
                idx += 1

            #del dataset
            self.dataset['input_ids'] = []

            print("Number of examples in training batch n" + str(step)+" : " + str(len(X)))

            return X,y

def set_seed(args,n_gpu):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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

def evaluate(args, model, eval_dataloader, wi, prefix=""):
    # Validation
    eval_output_dir = args.output_dir

    results = {} 

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    num_eval_examples = int(1653*0.2)
    logger.info("  Num examples = %d", num_eval_examples)
    logger.info("  Batch size = %d", 8)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    tweet_batch = TweetBatch(args.discretization_unit, args.window_size)
    n_batch = 1

    eval_iterator = tqdm(eval_dataloader, desc="Evaluating")

    for step, batch in enumerate(eval_iterator):
        # Set our model to evaluation mode (as opposed to training mode) to evaluate loss on validation set
        model = model.eval()         

        tweet_batch.discretize_batch(batch, step+1, n_batch)
        n_batch += 1

        X, y = tweet_batch.sliding_window(wi, device, step+1)

        # Forward pass
        if len(X)>=1: #the batch must contain, at least, one example, otherwise don't do forward  
            with torch.no_grad(): #in evaluation we tell the model not to compute or store gradients, saving memory and speeding up validation
                outputs = model(input_ids = X, labels=torch.tensor(y).to(device), weights=wi, window_size=args.window_size)   

                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = y
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, y, axis=0)
    
    eval_loss = eval_loss / nb_eval_steps

    preds = np.squeeze(preds) #because we are doing regression, otherwise it would be np.argmax(preds, axis=1)
    
    #since we are doing regression, our metric will be the mse
    result = mean_squared_error(preds, out_label_ids) #compute_metrics(eval_task, preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results
                
def main():
    parser = argparse.ArgumentParser(description='Finetune Bert for a regression task, to predict the tweet counts from the embbedings.')
    
    parser.add_argument('--dataset_path', default='bitcoin_data', help="OS path to the folder where the input ids are located.")
    parser.add_argument('--discretization_unit', default=1, help="The discretization unit is the number of hours to discretize the time series data. E.g.: If the user choses 3, then one sample point will cointain 3 hours of data.")
    parser.add_argument('--window_size', default=3, help="Number of time windows to look behind. E.g.: If the user choses 3, when to provide the features for the current window, we average the embbedings of the tweets of the 3 previous windows.")
    parser.add_argument("--save_steps", type=int, default=300, help="Save checkpoint every X updates steps.") 
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Warmup is the proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%")
    parser.add_argument("--model_name_or_path", default=r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\finetuning_outputs\checkpoint-10', type=str, help="Path to folder containing saved checkpoints, schedulers, models, etc.")
    parser.add_argument("--output_dir", default='finetuning_outputs', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_train_epochs", default=4, type=int, help="Total number of training epochs to perform." )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.")
    args = parser.parse_args()
    print(args) 

    tb_writer = SummaryWriter()

    dataset_path = cpath.joinpath(args.dataset_path)

    discretization_unit = args.discretization_unit
    window_size = args.window_size

    #Calculates the timedifference
    timedif = [i for i in range(window_size)]

    #Calculate the weights using K = 0.5 (giving 50% of importance to the most recent timestamp)
    #and tau = 6.25s so that when the temporal difference is 10s, the importance is +- 10.1%
    wi = weights(0.5, 2, timedif)

    # Approximation of batch_size
    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32, which represent 32 sample points (features, embbedings) after preprocessing is done
    batch_size_ = 1#32 
    approx = 45 #~ number of tweets in one hour
    batch_size = round(batch_size_+window_size*discretization_unit*approx-1)

    #load the dataset: timestamps and input ids (which correspond to the tweets already tokenized using BertTokenizerFast)
    #each chunk is read as a different dataset, and in the end all datasets are concatenated. A sequential sampler is defined.
    train_dataloader, dev_dataloader, test_dataloader = load_inputids(path = dataset_path, batch_size=batch_size)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    torch.save(test_dataloader, args.output_dir+'/test_dataloader.pth')

    num_train_examples = int(1653*0.8)

    #batch_size = 5 #approximate

    #Number of times the gradients are accumulated before a backward/update pass
    gradient_accumulation_steps = args.gradient_accumulation_steps

    # Number of training epochs, for finetuning in a specific NLP task, the authors recommend 2-4 epochs only
    epochs = args.num_train_epochs

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

        n_gpu = 1

        model.cuda()

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

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_train_examples)
    logger.info("  Num Epochs = %d", epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_train_optimization_steps)

    # Store our loss and accuracy for plotting
    train_loss_set = []

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

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

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()

    # trange is a tqdm wrapper around the normal python range
    train_iterator = trange(
        epochs_trained, epochs, desc="Epoch",
    )

    set_seed(args, n_gpu)  # Added here for reproductibility
    
    #Training
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration") 
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        tweet_batch = TweetBatch(discretization_unit, window_size)
        n_batch = 1

        # Train the data for one epoch
        for step, batch in enumerate(epoch_iterator):  

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            # Set our model to training mode (as opposed to evaluation mode)
            model = model.train()

            tweet_batch.discretize_batch(batch, step+1, n_batch)
            n_batch += 1

            X, y = tweet_batch.sliding_window(wi, device, step+1)

            # Clear out the gradients (by default they accumulate)
            #optimizer.zero_grad() #DUVIDA: ISTO E PARA TIRAR?

            # Forward pass
            if len(X)>=1: #the batch must contain, at least, one example, otherwise don't do forward
                
                loss, logits = model(input_ids = X, labels=torch.tensor(y).to(device), weights=wi, window_size=window_size)   

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                a = list(model.parameters())[199].clone()
                a2 = list(model.parameters())[200].clone()

                # Backward pass
                loss.backward()

                #Store 
                train_loss_set.append(loss.item()) 
                
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += len(X) #b_input.size(0)
                nb_tr_steps += 1

                print("Train loss: {}".format(tr_loss/nb_tr_steps))

                if (step + 1) % args.gradient_accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
               
                    # Update parameters and take a step using the computed gradient
                    optimizer.step()
                    scheduler.step() #update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    
                    if args.logging_steps>0 and global_step%args.logging_steps==0:
                        logs={}

                        if args.evaluate_during_training: 
                            results = evaluate(args, model, dev_dataloader, wi)
                            for key, value in results.items():
                                eval_key = "eval_{}".format(key)
                                logs[eval_key] = value

                        # DUVIDA: Pedir à zita para explicar isto
                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

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
                    if step%args.logging_steps==0:
                        logger.info("Check if the classifier layer weights are being updated:")
                        logger.info("Weight W: "+str(not torch.equal(a.data, b.data)))  
                        logger.info("Bias b: " + str(not torch.equal(a2.data, b2.data)))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()                

    #return global_step, tr_loss/global_step  

    plt.figure(figsize=(15,8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(tr_loss)
    plt.show()

if __name__ == '__main__':
    main()
    
