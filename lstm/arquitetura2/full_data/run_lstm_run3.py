# script para testar lstm com dataset grande p/ janela dt=1h
import lstm_run3 as run
import argparse
import os
import pdb

def main():
    parser = argparse.ArgumentParser(description='Fit an LSTM to the data, to predict the tweet counts from the embbedings.')
    parser.add_argument('--discretization_unit', default=1, help="The discretization unit is the number of hours to discretize the time series data. E.g.: If the user choses 3, then one sample point will cointain 3 hours of data.")
    parser.add_argument('--window_size', type = int, default=0, help='The window length defines how many units of time to look behind when calculating the features of a given timestamp.')
    parser.add_argument('--seq_len', type = int, default=24, help='Input dimension (number of timestamps).')
    parser.add_argument('--batch_size', type = int, default=1, help='How many batches of sequence length inputs per iteration.')
    parser.add_argument("--save_steps", type=int, default=100000, help="Save checkpoint every X updates steps.") 
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for Adam.") #5e-5
    parser.add_argument("--model_name_or_path", default=r'C:/Users/Filipa/Desktop/Predtweet/lstm/arquitetura2/sem_sliding_batches/lstm/fit_results/checkpoint-250/', type=str, help="Path to folder containing saved checkpoints, schedulers, models, etc.")    #r'C:/Users/Filipa/Desktop/Predtweet/bitcoin_data/TF-IDF/server/n_features/768/1.0/lstm/fit_results/checkpoint-200/', type=str, help="Path to folder containing saved checkpoints, schedulers, models, etc.")
    parser.add_argument("--num_train_epochs", default=300, type=int, help="Total number of training epochs to perform." )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--evaluate_during_training", action="store_false", help="Run evaluation during training at each logging step.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--test_acc", action="store_false", help="Run evaluation and store accuracy on test set.")
    parser.add_argument("--evaluate_only", action="store_true", help="Run only evaluation on validation and test sets with the best model found in training.")
    parser.add_argument("--use_features", action="store_true", help="If we want to consider the textual features (from BERT/TFIDF) or only the counts.")
    args = parser.parse_args()

    #loop_percentages = [[0.9,0.05,0.05], [0.8, 0.1, 0.1], [0.70, 0.15, 0.15], [0.60, 0.20, 0.20], [0.50, 0.25, 0.25]]
    loop_percentages = [[0.9, 0.05, 0.05], [0.8, 0.05, 0.05], [0.7, 0.05, 0.05], [0.6, 0.05, 0.05], [0.5, 0.05, 0.05]]
    loop_dt = [1]
    loop_dw = [0,1,3,5,7]
    
    for dt in loop_dt:
        for dw in loop_dw:
            n_run=1
            for percentages in loop_percentages:
                args.percentages = percentages
                if dw==0:
                    aux_path = '/mnt/hdd_disk2/frente/results/dt/'
                else:
                    aux_path = '/mnt/hdd_disk2/frente/results/'
                args.full_dataset_path = aux_path+str(dt)+'.'+str(dw)+'/'
                
                output_dir = '/mnt/hdd_disk2/frente/lstm_fit_results/arquitetura2/full_data/'+str(dt)+'.'+str(dw)+'/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_dir = output_dir+'run'+ str(n_run)+'/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                args.output_dir = output_dir
                
                run.main(args)
                n_run+=1

if __name__=='__main__':
     main()    
