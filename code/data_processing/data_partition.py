# Just using this to construct folds
# Actual experiments will be run with experiments.py

import os, json
import numpy as np
import pandas as pd
from itertools import chain
from random import shuffle

# Change directory to where data is kept and read it in
os.chdir('/Users/kylefth/Documents/mooc_project/Annotations')
ml_data = '/Users/kylefth/Documents/mooc_project/mooc_forum_mining/train_test_pairs'

df = pd.read_csv('latest_and_greatest_reduced.csv')
df.pop('id')
data = df.to_json(orient='records')
data = json.loads(data)

print data[-1].keys()

print
print
print

labels = [i['label'] for i in data]

for i in set(labels):
    print str(i) + ": " + str(labels.count(i))
    print str(i) + ": " + str(float(labels.count(i)) / float(len(labels))*100) + "%"
    print 

# Define function to partition data into equal-sized folds
def k_fold_chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.
    
    while last < len(seq):
        out.append(seq[int(last):int(last+avg)])
        last += avg
    for i,j in enumerate(out):
        with open("{0}split.json".format(i), 'w') as outfile:
            json.dump(j, outfile)
            
# Create function for partitioning data into train-test splits
os.chdir(ml_data)

# Randomize order of intstances
shuffle(data)   
# Partition data into 10 folds
k_fold_chunk(data, 10)
files = os.listdir(os.getcwd())
folds = []
for i in files:
    if i[-4:] == "json":
        with open(i, 'rb') as infile:
            data = json.load(infile)
            folds.append(data)
# Assemble train-test pairs
# Need to do this iteratively for each pair in
# the train-test pairs list below
pairs = []
for i in folds:
    test = i
    training = [j for j in folds if j != test]
    training = list(chain.from_iterable(training))
    pairs.append([training, test])            

for i,j in enumerate(pairs):
    with open("{0}traintestpair.json".format(i), 'w') as outfile:
        json.dump(j, outfile)

