import os
import numpy as np

ensemble_paths = [
    "results/run0/logit.txt",
    "results/run3/logit.txt"
]

save_path = "answer_ensemble.txt"

class_0_logit_collection = []
class_1_logit_collection = []

for file in ensemble_paths:
    logits_0 = []
    logits_1 = []
    with open(file, 'r+') as fin:
        for line in fin:
            logits_0.append(float(line.strip().split(',')[0]))
            logits_1.append(float(line.strip().split(',')[1]))
    if not len(class_0_logit_collection) == 0:
        assert len(class_0_logit_collection[0]) == len(logits_0)
        assert len(class_1_logit_collection[0]) == len(logits_1)
    class_0_logit_collection.append(np.array(logits_0).astype('float32'))
    class_1_logit_collection.append(np.array(logits_1).astype('float32'))

class_0_logit_avg = np.average(np.array(class_0_logit_collection), axis=0)
class_1_logit_avg = np.average(np.array(class_1_logit_collection), axis=0)

print(class_0_logit_avg)
print(class_1_logit_avg)

with open(save_path, 'w') as answer_file:
    for i in range(class_0_logit_avg.shape[0]):
        if class_0_logit_avg[i] > class_1_logit_avg[i]:
            label = 'NOT_SARCASM'
        else:
            label = 'SARCASM'
        answer_file.write("twitter_{},{}".format(i + 1, label))
        answer_file.write('\n')
