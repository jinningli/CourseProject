# Text Classification Competition: Twitter Sarcasm Detection

## Evaluate Model and Ensemble

#### Download pre-trained model for our method

Download `checkpoint.zip` from https://drive.google.com/file/d/1nRucz1yDqyoYR8jLeP6fKZt8FpqJBBUA/view?usp=sharing
```
unzip checkpoint.zip
mkdir checkpoint
mkdir checkpoint/run0
mv checkpoint checkpoint/run0/checkpoint
```
#### Evaluate and Generating Predictions
```
python3 evaluate.py --run run0 --use_bert --device 3
```

#### Ensemble
```
# modify ensemble_paths in ensemble.py
python3 ensemble.py
```

#### All Parameters
`--run` Evaluating on runX. e.g. --run=run1

`--use_bert` Use bert model. If not, using LSTM model.

`--device` GPU device to use

## Train Model

#### Start Training with Validation and Evaluation
```
python3 main.py --use_bert --device 2 --lr 2e-5 --epochs 20 --use_valid --eval
```

#### Start Training without Validation
```
python3 main.py --use_bert --device 2 --lr 2e-5 --epochs 20
```

#### Start Training using LSTM model
```
python3 main.py --device 2 --lr 2e-5 --epochs 20
```

#### All Parameters
`--epochs` Number of epochs for training

`--batch_size` Batch size

`--lr` Learning rate for optimizer

`--run` Continue training on runX. Eg. --run=run1

`--eval` Evaluate on training set

`--use_valid` Use valid dataset. If not, using the whole dataset for training

`--use_bert` Use bert model. If not, using the LSTM model

`--split_ratio` When using validation, percentage of trainset

`--device` GPU device to use

## Appendix: Competetion Data Format

Each line contains a JSON object with the following fields :
- ***response*** :  the Tweet to be classified
- ***context*** : the conversation context of the ***response***
	- Note, the context is an ordered list of dialogue, i.e., if the context contains three elements, `c1`, `c2`, `c3`, in that order, then `c2` is a reply to `c1` and `c3` is a reply to `c2`. Further, the Tweet to be classified is a reply to `c3`.
- ***label*** : `SARCASM` or `NOT_SARCASM`

- ***id***:  String identifier for sample. This id will be required when making submissions. (ONLY in test data)

For instance, for the following training example :

`"label": "SARCASM", "response": "@USER @USER @USER I don't get this .. obviously you do care or you would've moved right along .. instead you decided to care and troll her ..", "context": ["A minor child deserves privacy and should be kept out of politics . Pamela Karlan , you should be ashamed of your very angry and obviously biased public pandering , and using a child to do it .", "@USER If your child isn't named Barron ... #BeBest Melania couldn't care less . Fact . 💯"]`

The response tweet, "@USER @USER @USER I don't get this..." is a reply to its immediate context "@USER If your child isn't..." which is a reply to "A minor child deserves privacy...". Your goal is to predict the label of the "response" while optionally using the context (i.e, the immediate or the full context).

***Dataset size statistics*** :

| Train | Test |
|-------|------|
| 5000  | 1800 |

For Test, we've provided you the ***response*** and the ***context***. We also provide the ***id*** (i.e., identifier) to report the results.

***Submission Instructions*** : Follow the same instructions as for the MPs -- create a private copy of this repo and add a webhook to connect to LiveDataLab.Please add a comma separated file named `answer.txt` containing the predictions on the test dataset. The file should have no headers and have exactly 1800 rows. Each row must have the sample id and the predicted label. For example:

twitter_1,SARCASM
twitter_2,NOT_SARCASM
...
