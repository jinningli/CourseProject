import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import joblib
from scipy.stats import entropy
import matplotlib.pyplot as plt
import statistics
import math
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from dataset import Dataset
from model import SimpleLSTMBaseline, BERT
from torchtext.data import Iterator, BucketIterator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='', help='Continue training on runX. Eg. --run=run1')
    parser.add_argument('--use_bert', action='store_true', default=False, help='Use bert model')
    parser.add_argument('--device', type=str, default="0", help='GPU device to use')
    args = parser.parse_args()
    args.result_path = "results/"
    args.checkpoint_dir = "checkpoint/"
    args.checkpoint = "checkpoint"
    args.dataset = "data/"
    args.device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'
    args.seed = 1337
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.run) == 0:
        run_count = len([dir for dir in os.listdir(args.checkpoint_dir) if dir[0:3] == "run"])
        args.run = 'run{}'.format(run_count-1)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run)
    args.result_path = os.path.join(args.result_path, args.run)+'/'
    os.makedirs(args.result_path, exist_ok=True)
    return args

def run_evaluation(args, checkpoint, test_loader, vocab_size):
    device = args.device
    model = SimpleLSTMBaseline(vocab_size, hidden_dim=128, emb_dim=128).to(device)
    # model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    answer_file = open(args.result_path+'/answer.txt', "w")
    for i, batch in enumerate(test_loader):
        text, context = batch.text.to(device), batch.context.to(device)
        pred = model(text)
        if F.sigmoid(pred) >= 0.5:
            label = 'SARCASM'
        else:
            label = 'NOT_SARCASM'
        answer_file.write("twitter_{},{}".format(i+1, label))
        answer_file.write('\n')
    answer_file.close()

def run_evaluation_bert(args, checkpoint, test_loader, vocab_size):
    device = args.device
    model = BERT().to(device)
    # model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    answer_file = open(args.result_path+'/answer.txt', "w")
    # For ensemble
    logit_file = open(args.result_path + '/logit.txt', "w")
    for i, batch in enumerate(test_loader):
        text, context = batch.text, batch.context
        text = text.type(torch.LongTensor).to(device)
        output = model.run_eval(text)
        pred = torch.argmax(output, 1).tolist()
        assert len(pred) == 1
        if pred[0] == 1:
            label = 'SARCASM'
        elif pred[0] == 0:
            label = 'NOT_SARCASM'
        else:
            raise NotImplementedError("Strange pred.")
        answer_file.write("twitter_{},{}".format(i+1, label))
        answer_file.write('\n')
        logit_file.write("{},{}".format(output[0][0], output[0][1]))
        logit_file.write("\n")
    answer_file.close()
    logit_file.close()

def main(args):
    tic = time.time()
    print(args.run)
    train_dataset, _, test_dataset, vocab_size = Dataset(train_jsonl_file=args.dataset+'train.jsonl', test_jsonl_file=args.dataset+'test.jsonl', use_bert=args.use_bert)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device))
    test_loader = Iterator(test_dataset, batch_size=1, shuffle=False)
    if args.use_bert:
        average = run_evaluation_bert(args, checkpoint, test_loader, vocab_size)
    else:
        average = run_evaluation(args, checkpoint, test_loader, vocab_size)
    print('[{:.2f}] Finish evaluation {}'.format(time.time() - tic, args.run))

if __name__ == '__main__':
    args = get_args()
    main(args)
