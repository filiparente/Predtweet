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

#current path
cpath = Path.cwd()

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
    parser = argparse.ArgumentParser(description='Train a neural network for a regression task, to predict the tweet counts from the embbedings.')
    
    parser.add_argument('--dataset_path', default='results/3.1/', help="OS path to the folder where the dataset is located.")
    parser.add_argument('--dataset_name', default='dataset.txt', help="Name of the dataset, including file extension.")
    
    args = parser.parse_args()
    print(args) 

    dataset_path = cpath.joinpath(args.dataset_path)
    dataset_path = dataset_path.joinpath(args.dataset_name)

    #load the dataset: features are the embeddings and labels are the tweet counts
    features, labels, window_size = load_dataset(dataset_path)

    percentages = [0.8,0.1,0.1] #train, validation, test
    #test/train split
    # TODO
    train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels = train_test_split(features, labels, percentages, window_size)

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

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01}, #0.01
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0} #0.0
    ]
    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=2e-4, #2e-5
                        warmup=.0)

    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False
        
    # Convert all of our data into torch tensors, the required datatype for our model

    train_inputs = torch.tensor(train_inputs, dtype=float, device=device)
    validation_inputs = torch.tensor(validation_inputs, dtype=float, device=device)
    train_labels = torch.tensor(train_labels, dtype=float, device=device)
    validation_labels = torch.tensor(validation_labels, dtype=float, device=device)

    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
    batch_size = 100

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Store our loss and accuracy for plotting
    train_loss_set = []
    dev_acc_set = []

    # Number of training epochs 
    epochs = 200

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
    
        # Training   
        # Set our model to training mode (as opposed to evaluation mode)
        model = model.train()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, logits = model(inputs_embeds = b_input.float(), labels=b_labels.float())
            train_loss_set.append(loss.item())    
            #a = list(model.parameters())[199].clone()
            #a2 = list(model.parameters())[200].clone()

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            #b = list(model.parameters())[199].clone()
            #b2 = list(model.parameters())[200].clone()
            # PARA CONFIRMAR SE OS PESOS ESTAVAM A SER ALTERADOS OU NÃO: 199 é o peso W e 200 é o peso b (bias) da layer de linear de classificação/regressão: WX^T+b
            #print(torch.equal(a.data, b.data))
            #print(torch.equal(a2.data, b2.data))
            
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
            
            
        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model = model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(inputs_embeds = b_input.float())
                
            # Move logits and labels to CPU
            logits = logits[0].detach().cpu().numpy()
            #print(logits)
            label_ids = b_labels.to('cpu').numpy()

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


def train(x,y):
    torch.manual_seed(1)    # reproducible

    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(torch.tensor(x, dtype=float)), Variable(torch.tensor(y, dtype=float))

    # another way to define a network
    net = torch.nn.Sequential(
            torch.nn.Linear(768, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 1),
        )
    net.double()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 1
    EPOCH = 20

    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2,)


    # start training
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            
            b_x = Variable(batch_x)
            b_y = Variable(batch_y).reshape(1,1)

            prediction = net(b_x)     # input x and predict based on x

            loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
            print(loss)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

    return net


if __name__ == '__main__':
    main()
    