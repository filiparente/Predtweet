import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from transformers import BertForSequenceClassification
from run_glue import basic_train

#current path
cpath = Path.cwd()

def main():
    parser = argparse.ArgumentParser(description='Train a neural network for a regression task, to predict the tweet counts from the embbedings.')
    
    parser.add_argument('--dataset_path', default='bitcoin_data/', help="OS path to the folder where the dataset is located.")
    parser.add_argument('--dataset_name', default='bitcoin_dataset.txt', help="Name of the dataset, including file extension.")
    
    args = parser.parse_args()
    print(args) 

    dataset_path = cpath.joinpath(args.dataset_path)
    dataset_path = dataset_path.joinpath(args.dataset_name)

    #load the dataset: features are the embeddings and labels are the tweet counts
    features, labels = load_dataset(dataset_path)

    #test/train split
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

    #model = train(train_features, train_labels)

    #results = evaluate(test_features, test_labels)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            param.requires_grad = False
        
    #basic train without batches
    outputs = basic_train(model, train_features, train_labels)

    loss, logits = outputs[:2]



def load_dataset(dataset_path):
    with open(dataset_path,encoding='utf-8', errors='ignore', mode='r') as j:
        data = json.loads(j.read())

        features = [el['X'] for el in data]
        labels = [el['y'] for el in data]

        return features, labels


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
    