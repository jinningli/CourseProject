from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import json
import random
import torch

from transformers import BertTokenizer

def Dataset(train_jsonl_file, test_jsonl_file, use_valid=False, split_ratio=0.8, use_bert=False):
    """Create dataset with torchtext"""
    train_file = train_jsonl_file[:-len('jsonl')]+'json'
    with open(train_jsonl_file, 'rb') as file:
        my_data_file = open(train_file, 'w')
        for jline in file:
            item = json.loads(jline)
            # immediate context
            item['context'] = item['context'][-1]
            json.dump(item, my_data_file)
            my_data_file.write("\n")
    my_data_file.close()

    test_file = test_jsonl_file[:-len('jsonl')]+'json'
    with open(test_jsonl_file, 'rb') as file:
        my_data_file = open(test_file, 'w')
        for jline in file:
            item = json.loads(jline)
            item['context'] = item['context'][-1]
            json.dump(item, my_data_file)
            my_data_file.write("\n")
    my_data_file.close()

    if use_bert:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Model parameter
        MAX_SEQ_LEN = 128
        PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

        # Fields

        LABEL = Field(sequential=False, use_vocab=True, unk_token=None)
        TEXT = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False,
                           batch_first=True,
                           fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    else:
        TEXT = Field(sequential=True, lower=True)
        LABEL = Field(sequential=False, use_vocab=True, unk_token=None)

    fields = {'label': ('label', LABEL), 'response': ('text', TEXT), 'context': ('context', TEXT)}
    train_dataset = TabularDataset(
        path=train_file,
        format='JSON',
        fields=fields)
    LABEL.build_vocab(train_dataset)

    val_dataset = None
    if use_valid:
        train_dataset, val_dataset = train_dataset.split(split_ratio, random_state=random.seed(0))

    fields = {'response': ('text', TEXT), 'context': ('context', TEXT)}
    test_dataset = TabularDataset(
        path=test_file,
        format='JSON',
        fields=fields)
    TEXT.build_vocab(train_dataset, test_dataset)
    return train_dataset, val_dataset, test_dataset, len(TEXT.vocab)
