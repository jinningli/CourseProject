import warnings
warnings.filterwarnings("ignore")
import os
import copy
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib.animation as animation
import sklearn.metrics

from torchtext.data import BucketIterator
from dataset import Dataset
from model import SimpleLSTMBaseline, BERT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs for training (emixer default=400k)')
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size (emixer default=512)")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate for optimizer")
    parser.add_argument('--run', default='', help='Continue training on runX. Eg. --run=run1')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluate on training set')
    parser.add_argument('--use_valid', action='store_true', default=False, help='Use valid dataset')
    parser.add_argument('--use_bert', action='store_true', default=False, help='Use bert model')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Use valid dataset')
    parser.add_argument('--device', type=str, default="0", help='GPU device to use')
    args = parser.parse_args()
    args.dataset = "data/"
    args.checkpoint_path = "checkpoint/"
    args.checkpoint = "checkpoint"

    args.seed = 1337
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    if len(args.run) == 0:
        run_count = len([dir for dir in os.listdir(args.checkpoint_path) if dir[0:3] == "run"])
        args.run = 'run{}'.format(run_count)
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.run)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    return args

def run_training(args, dataset, train_loader, val_loader, vocab_size):
    checkpoint_path = os.path.join(args.checkpoint_path, args.checkpoint)
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:" + args.device if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    model = SimpleLSTMBaseline(vocab_size, hidden_dim=128, emb_dim=128).to(device)
    # Handle multi-gpu if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    #     print("Let's use", ngpu, "GPUs!")
    #     model = nn.DataParallel(model, list(range(ngpu)))
    # model.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCEWithLogitsLoss()
    # Setup Adam optimizers for both G and D
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train() # turn on training mode
    # Training Loop
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args.epochs):
        # For each batch in the dataloader
        losses = []
        running_corrects = 0
        for i, batch in enumerate(train_loader):
            # Format batch
            optimizer.zero_grad()
            text, context, label = batch.text.to(device), batch.context.to(device), batch.label.to(device)
            pred = model(text)
            loss = criterion(pred, label.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = sum(losses) / len(losses)
        print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, epoch_loss))
        # save model
        if epoch % 1 == 0 or epoch == args.epochs-1:
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab_size': vocab_size,
            'args': vars(args)
            }, checkpoint_path)
            if args.eval:
                preds = []
                labels = []
                for i, batch in enumerate(val_loader if val_loader is not None else train_loader):
                    text, context, label = batch.text.to(device), batch.context.to(device), batch.label.to(device)
                    pred = model(text)
                    pred = F.sigmoid(pred) >= 0.5
                    preds.extend(pred.cpu().tolist())
                    labels.extend(label.cpu().tolist())
                print("{} Precision: {}, Recall: {}, F1: {}".format(
                "Train" if val_loader is None else "Valid",
                sklearn.metrics.precision_score(np.array(labels).astype('int32'), np.array(preds)),
                sklearn.metrics.recall_score(np.array(labels).astype('int32'), np.array(preds)),
                sklearn.metrics.f1_score(np.array(labels).astype('int32'), np.array(preds))
                ))

def run_training_bert(args, dataset, train_loader, val_loader, vocab_size):
    checkpoint_path = os.path.join(args.checkpoint_path, args.checkpoint)
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")

    model = BERT().to(device)

    # Initialize BCELoss function
    # criterion = nn.BCEWithLogitsLoss()
    # Setup Adam optimizers for both G and D
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    model.train() # turn on training mode
    # Training Loop
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args.epochs):
        # For each batch in the dataloader
        losses = []
        running_corrects = 0
        for i, batch in enumerate(train_loader):
            # format batch
            text, context, label = batch.text, batch.context, batch.label
            # print(text.tolist()[0])
            # print(label.tolist()[0])
            label = label.type(torch.LongTensor).to(device)
            text = text.type(torch.LongTensor).to(device)

            output = model(text, label)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        epoch_loss = sum(losses) / len(losses)
        print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, epoch_loss))
        # save model
        if epoch % 1 == 0 or epoch == args.epochs-1:
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab_size': vocab_size,
            'args': vars(args)
            }, checkpoint_path)
            if args.eval:
                model.eval()
                with torch.no_grad():
                    preds = []
                    labels = []
                    eval_losses = []
                    for i, batch in enumerate(val_loader if val_loader is not None else train_loader):
                        text, context, label = batch.text, batch.context, batch.label
                        label = label.type(torch.LongTensor).to(device)
                        text = text.type(torch.LongTensor).to(device)
                        output = model(text, label)
                        loss, output = output
                        pred = torch.argmax(output, 1).tolist()
                        preds.extend(pred)
                        labels.extend(label.tolist())
                        eval_losses.append(loss.item())
                    print("{} Precision: {}, Recall: {}, F1: {}, Loss: {}".format(
                    "Train" if val_loader is None else "Valid",
                    sklearn.metrics.precision_score(np.array(labels).astype('int32'), np.array(preds)),
                    sklearn.metrics.recall_score(np.array(labels).astype('int32'), np.array(preds)),
                    sklearn.metrics.f1_score(np.array(labels).astype('int32'), np.array(preds)),
                    np.average(eval_losses)
                    ))

def main(args):
    global tic
    tic = time.time()
    print("dataset processing ...")
    train_dataset, valid_dataset, test_dataset, vocab_size = Dataset(train_jsonl_file=args.dataset+'train.jsonl', test_jsonl_file=args.dataset+'test.jsonl', use_valid=args.use_valid, split_ratio=args.split_ratio, use_bert=args.use_bert) # "train.csv" "test.csv" "validate.csv"
    print("train_dataset len:", len(train_dataset))
    print("epochs:{} batch:{} lr:{}".format(args.epochs, args.batch_size, args.lr))
    print("run:{} device:{} gpu count:{} cpu count:{}".format(args.run, args.device, torch.cuda.device_count(), os.cpu_count())) # , shuffle=True
    train_loader = BucketIterator(train_dataset, batch_size=args.batch_size, sort_key=lambda x: len(x.text), sort_within_batch=False)
    print("train_loader len:", len(train_loader))
    valid_loader = None
    if args.use_valid:
        valid_loader = BucketIterator(valid_dataset, batch_size=args.batch_size, sort_key=lambda x: len(x.text), sort_within_batch=False)
        print("valid_loader len:", len(valid_loader))
    if args.use_bert:
        run_training_bert(args, train_dataset, train_loader, valid_loader, vocab_size)
    else:
        run_training(args, train_dataset, train_loader, valid_loader, vocab_size)
    print('[{:.2f}] Finish training {}'.format(time.time() - tic,  args.run))

if __name__ == '__main__':
    args = get_args()
    main(args)
