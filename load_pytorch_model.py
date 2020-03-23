import torch
from finetuning import TweetBatch, weights
from tqdm import tqdm, trange
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
import pdb
from transformers import BertForSequenceClassification

def evaluate(args, model, eval_dataloader, wi, device, prefix=""):
    # Validation
    eval_output_dir = args.output_dir

    results = {} 

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    
    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    num_eval_examples = int(1653*0.2)
    print("  Num examples = %d", num_eval_examples)
    print("  Batch size = %d", 8)

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
                pdb.set_trace()
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
        print("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            print("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results

parser = argparse.ArgumentParser(description='Test Bert finetuned for a regression task, to predict the tweet counts from the embbedings.')
parser.add_argument('--discretization_unit', default=1, help="The discretization unit is the number of hours to discretize the time series data. E.g.: If the user choses 3, then one sample point will cointain 3 hours of data.")
parser.add_argument('--window_size', default=3, help="Number of time windows to look behind. E.g.: If the user choses 3, when to provide the features for the current window, we average the embbedings of the tweets of the 3 previous windows.")
parser.add_argument("--output_dir", default='/bitcoin_data/test_results1', type=str, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--model_path", default=r"C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\pytorch_model.bin", type=str)
parser.add_argument("--dataset_path", default=r"C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\finetuning_outputs\test_dataloader.pth", type=str )
args = parser.parse_args() 

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

model.load_state_dict(torch.load(args.model_path))

print("Done!")

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

    n_gpu = 1

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    

#Load test dataloader
test_dataloader = torch.load(args.dataset_path) 

#Calculates the timedifference
timedif = [i for i in range(args.window_size)]

#Calculate the weights using K = 0.5 (giving 50% of importance to the most recent timestamp)
#and tau = 6.25s so that when the temporal difference is 10s, the importance is +- 10.1%
wi = weights(0.5, 2, timedif)

results = evaluate(args, model, test_dataloader, wi, device)