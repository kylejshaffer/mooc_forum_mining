"""
Created on Sat Jan 10 17:59:58 2015

@author: kylefth
"""

import os, json, math
import numpy as np
import pandas as pd
from random import shuffle
from itertools import chain

# Change directory to where data is kept and read it in
feature_data = '/Users/kylefth/Dropbox/icwsm2015/features'
label_data = '/Users/kylefth/Documents/mooc_project/mooc_jaime_analysis/labels'
train_test_pairs = '/Users/kylefth/Documents/mooc_project/mooc_forum_mining/train_test_pairs'
os.chdir(feature_data)

# LIWC affective features
affective = pd.read_csv('affective.tsv', sep='\t')
del affective['Unnamed: 9']
# Binary indicator as to whether post is written by instroctor or not
author = pd.read_csv('author.tsv', sep='\t')
all_features = affective.merge(author, on='ID')
# LIWC Cognitive engagement features
cognitive = pd.read_csv('cognitive.tsv', sep='\t')
del cognitive['Unnamed: 10']
all_features = all_features.merge(cognitive, on='ID')
# Context features (need to read lines due to parsing error)
context = open('context.tsv').readlines()
context = [i.split('\t') for i in context]
for i,j in enumerate(context):
    if len(j) > 8: 
        context[i] = j[:-1]
    j[-1] = j[-1].strip('\r\n')

context = pd.DataFrame(context, columns=context[0])
context = context.ix[1:]
context = context.applymap(float)
context.ID = context.ID.apply(int)
all_features = all_features.merge(context, on='ID')
# Cosine similarity features
cosine = pd.read_csv('cosine.tsv', sep='\t')
all_features = all_features.merge(cosine, on='ID')
# Concern LIWC Features
concerns = pd.read_csv('current_concerns.tsv', sep='\t')
del concerns['Unnamed: 8']
all_features = all_features.merge(concerns, on='ID')
# Deadline manually constructed features
deadline = pd.read_csv('deadline.tsv', sep='\t')
all_features = all_features.merge(deadline, on='ID')
# Linguistic LIWC features
linguistic = pd.read_csv('linguistic.tsv', sep='\t')
del linguistic['Unnamed: 27']
all_features = all_features.merge(linguistic, on='ID')
# Link features
links = pd.read_csv('links.tsv', sep='\t')
all_features = all_features.merge(links, on='ID')
# Modal verbs
modals = pd.read_csv('modal.tsv', sep='\t')
all_features = all_features.merge(modals, on='ID')
# LIWC percpetual features
perceptual = pd.read_csv('perceptual.tsv', sep='\t')
del perceptual['Unnamed: 5']
all_features = all_features.merge(perceptual, on='ID')
# Position features
position = pd.read_csv('position.tsv', sep='\t')
all_features = all_features.merge(position, on='ID')
# Post vs comment features
post_comment = pd.read_csv('post_comment.tsv', sep='\t')
all_features = all_features.merge(post_comment, on='ID')
# Punctuation features
punctuation = pd.read_csv('punctuation.tsv', sep='\t')
del punctuation['Unnamed: 14']
all_features = all_features.merge(punctuation, on='ID')
# Seniment features
sentiment = pd.read_csv('sentiment.tsv', sep='\t')
all_features = all_features.merge(sentiment, on='ID')
# LIWC Social Features
social = pd.read_csv('social.tsv', sep='\t')
del social['Unnamed: 5']
all_features = all_features.merge(social, on='ID')
# LIWC spoken features
spoken = pd.read_csv('spoken.tsv', sep='\t')
del spoken['Unnamed: 3']
all_features = all_features.merge(spoken, on='ID')
all_features = all_features.reindex(np.random.permutation(all_features.index))

# Read in labels and convert to binary
labels = pd.read_csv(str(label_data) + '/labels.tsv', sep='\t')
# Convert labels to binary
label_list = list(labels.columns)
np_labels = np.array(labels)
np_id = np.array(labels.ID)
np_labels[np_labels < 3] = 0
np_labels[np_labels > 2] = 1
labels = pd.DataFrame(np_labels, columns=label_list)
labels.ID = np_id

# Function for partitioning data into folds
def fold_partition(df):
    cols = list(df.columns)
    np_features_split = np.array_split(df, 10)
    for i,j in enumerate(np_features_split):
        fold = pd.DataFrame(j, columns=cols)
        fold.to_csv('{0}split.csv'.format(i))

# Unigram features
# Create partition for each speech act and associated unigrams
# ANSWER
unigram_answer = pd.read_csv('unigram.answer.tsv', sep='\t')
del unigram_answer['Unnamed: 101']
unigram_answer = unigram_answer.merge(all_features, on='ID')
label = labels[['ID', 'A']]
label.columns = ['ID', 'LABEL']
unigram_answer = unigram_answer.merge(label, on='ID')
os.chdir(train_test_pairs + '/answer')
fold_partition(unigram_answer)
# ISSUE
os.chdir(feature_data)
unigram_issue = pd.read_csv('unigram.issue.tsv', sep='\t')
del unigram_issue['Unnamed: 101']
unigram_issue = unigram_issue.merge(all_features, on='ID')
label = labels[['ID', 'I']]
label.columns = ['ID', 'LABEL']
unigram_issue = unigram_issue.merge(label, on='ID')
os.chdir(train_test_pairs + '/issue')
fold_partition(unigram_issue)
# ISSUE RESOLUTION
os.chdir(feature_data)
unigram_issue_resolution = pd.read_csv('unigram.issue_resolution.tsv', sep='\t')
del unigram_issue_resolution['Unnamed: 101']
unigram_issue_resolution = unigram_issue_resolution.merge(all_features, on='ID')
label = labels[['ID', 'R']]
label.columns = ['ID', 'LABEL']
unigram_issue_resolution = unigram_issue_resolution.merge(label, on='ID')
os.chdir(train_test_pairs + '/issue_resolution')
fold_partition(unigram_issue_resolution)
# NEGATIVE ACKNOWLEDGEMENT
os.chdir(feature_data)
unigram_negative = pd.read_csv('unigram.negative_ack.tsv', sep='\t')
del unigram_negative['Unnamed: 101']
unigram_negative = unigram_negative.merge(all_features, on='ID')
label = labels[['ID', 'N']]
label.columns = ['ID', 'LABEL']
unigram_negative = unigram_negative.merge(label, on='ID')
os.chdir(train_test_pairs + '/negative_ack')
fold_partition(unigram_negative)
# OTHER
os.chdir(feature_data)
unigram_other = pd.read_csv('unigram.other.tsv', sep='\t')
del unigram_other['Unnamed: 101']
unigram_other = unigram_other.merge(all_features, on='ID')
label = labels[['ID', 'O']]
label.columns = ['ID', 'LABEL']
unigram_other = unigram_other.merge(label, on='ID')
os.chdir(train_test_pairs + '/other')
fold_partition(unigram_other)
# POSITIVE ACKNOWLEDGEMENT
os.chdir(feature_data)
unigram_positive = pd.read_csv('unigram.positive_ack.tsv', sep='\t')
del unigram_positive['Unnamed: 101']
unigram_positive = unigram_positive.merge(all_features, on='ID')
label = labels[['ID', 'P']]
label.columns = ['ID', 'LABEL']
unigram_positive = unigram_positive.merge(label, on='ID')
os.chdir(train_test_pairs + '/positive_ack')
fold_partition(unigram_positive)
# QUESTION
os.chdir(feature_data)
unigram_question = pd.read_csv('unigram.question.tsv', sep='\t')
del unigram_question['Unnamed: 101']
unigram_question = unigram_question.merge(all_features, on='ID')
label = labels[['ID', 'Q']]
label.columns = ['ID', 'LABEL']
unigram_question = unigram_question.merge(label, on='ID')
os.chdir(train_test_pairs + '/question')
fold_partition(unigram_question)


# Still need to generate train-test pairs from data
label_list = ['question', 'answer', 'issue', 'issue_resolution', 'positive_ack', 'negative_ack', 'other']
for lab in label_list:
    os.chdir(train_test_pairs + '/' + lab)
    folds = []
    files = os.listdir(os.getcwd())
    for f in files:
        if f[-4:] == '.csv':
            df = pd.read_csv(f)
            del df['Unnamed: 0']
            data = df.to_json(orient='records')
            data = json.loads(data)
            folds.append(data)
    pairs = []
    for i in folds:
        test = i
        training = [j for j in folds if j != test]
        training = list(chain.from_iterable(training))
        pairs.append([training, test])            
    
    os.chdir('pairs')
    for i,j in enumerate(pairs):
        with open("{0}traintestpair.json".format(i), 'w') as outfile:
            json.dump(j, outfile)