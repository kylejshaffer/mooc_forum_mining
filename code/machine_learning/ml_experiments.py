'''
Initial script for running machine learning experiments using K-fold
cross validation. Results from these experiments are not reported 
in the paper. 
'''

import os, json, math
import numpy as np
import pandas as pd
from random import shuffle
from sklearn import preprocessing, metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform as sp_uniform
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Change directory to where data is kept and read it in
ml_data = '/Users/kylefth/Documents/mooc_project/mooc_forum_mining/train_test_pairs'

# Function to construct train-test pairs a bit more
# dynamically depending on whether you want to do
# 10-fold or 20-fold cross validation
def train_test_pairs(folds):
    if folds == 10:
        os.chdir(ml_data + "/10_fold")
    elif folds == 20:
        os.chdir(ml_data + "/20_fold")

    files = os.listdir(os.getcwd())

    pairs = []
    for i in files:
        if i[-6:] == 'r.json':
            with open(i, 'rb') as infile:
                data = json.load(infile)
                pairs.append(data)
    return pairs         

# Function to run experiments on any label using any algorithm that is passed in
def run_cross_val(label, alg, folds):
    pairs = train_test_pairs(folds)
    avg_prec_scores = []
    for p in pairs:
        current_data = p
        # Test set, test set
        X = current_data[0]
        y = current_data[1]
        
        for i in X:
            if i['label'] != str(label):
                i['label'] = 'na'
                
        for i in y:
            if i['label'] != str(label):
                i['label'] = 'na'
                
        train_labs = [i['label'] for i in current_data[0]]
        test_labs = [i['label'] for i in current_data[1]]
        
        print "Training labels are %s" % set(train_labs)
        print
        print "Test labels are %s" % set(test_labs)
        print
        
        # Evening out training data for 50/50 split
        pos_class = [i for i in X if i['label'] == str(label)]
        neg_class = [i for i in X if i['label'] != str(label)]
        print "There are %s instances of the positive class" % len(pos_class)
        print
        print "There are %s instances of the negative class" % len(neg_class)
        print
        
        # Deal with ratios as floats at first then take floor or ceiling division
        # Run this and look at ratio calculations and make sure these are right
        if len(neg_class) > len(pos_class):
            ratio = int(math.ceil(float(len(neg_class)) / float(len(pos_class))))
            print "Original class ratio is %s: 1, negative_class: positive_class" % ratio
            pos_class = pos_class * ratio
            print "The positive class now contains %s instances" % len(pos_class)

        elif len(pos_class) > len(neg_class):
            ratio = int(math.ceil(float(len(pos_class)) / float(len(neg_class))))
            print "Original class ratio is %s: 1, positive_class: negative_class" % ratio
            neg_class = neg_class * ratio
            print "The negative class now contains %s instances" % len(neg_class)
            
        X = pos_class + neg_class
        print
        print "Your training set now has %s instances in it" % len(X)
        
        while X[0]['label'] != str(label):
            shuffle(X)
                
        train_labels = [i['label'] for i in X]
        test_labels = [i['label'] for i in y]
        
        # Create vector of issue labels for sklearn classifiers
        # Class of interest is always encoded as 1
        for i,j in enumerate(train_labels):
            if j == str(label):
                train_labels[i] = 1
            elif j == 'na':
                train_labels[i] = 0
        
        vec_train_labels = np.array(train_labels)
        
        for i,j in enumerate(test_labels):
            if j == str(label):
                test_labels[i] = 1
            elif j == 'na':
                test_labels[i] = 0
        
        vec_test_labels = np.array(test_labels)
        
        # Remove target labels from data instances
        X_nolab = X
        for i in X_nolab:
            if 'label' in i.keys():
                del i['label']
        
        y_nolab = y
        for i in y_nolab:
            del i['label']
        
        # Initialize vectorizer object to convert data instances to 
        # numpy format for sklearn
        vec = DictVectorizer()
        vec_train = vec.fit_transform(X_nolab).toarray()
        vec_test = vec.fit_transform(y_nolab).toarray()
        # list_data = [i.values() for i in data_nolabels]
        # np_data = np.array(list_data)
        # Min-max normalization
        scaler = preprocessing.MinMaxScaler()
        scaled_train_data = scaler.fit_transform(vec_train)
        scaled_test_data = scaler.transform(vec_test)
        # Chi-squared feature selection
        # Haven't used this yet, but may be useful for feature ablation
        # chi_pruned_data = SelectKBest(chi2, k=45).fit_transform(scaled_data, labels)
            
        # Initialize chosen classifier instance, train it on 
        # data, and generate predictions based on this model
        clf = alg
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}        
        grid_search = GridSearchCV(clf, param_grid)
        print "Training model..."
        grid_search.fit(scaled_train_data, vec_train_labels)
        print "Testing model..."
        predicted = grid_search.predict(scaled_test_data)
        
        # Print out metrics for evaluation
        print_metrics(vec_test_labels, predicted, avg_prec_scores)
    
    print
    mean_avg_prec = float(sum(avg_prec_scores)) / float(len(avg_prec_scores))
    print "Mean average precision over " + str(folds) + " folds: %s" % mean_avg_prec

# Function for printing metrics
def print_metrics(true_labels, predicted_labels, avg_prec_scores):
    avg_prec_scores = avg_prec_scores
    print "Held-out validation results"
    print "=" * 80
    print "CLASSIFICATION REPORT"
    print metrics.classification_report(true_labels, predicted_labels, target_names=['non-class', 'class'])
    print 
    print "Confusion Matrix"
    print metrics.confusion_matrix(true_labels, predicted_labels)
    print 
    print "AVERAGE PRECISION"
    print "Corresponds to area under PR curve"
    # Averge precision
    avg_prec = metrics.average_precision_score(true_labels, predicted_labels, average=None)
    print avg_prec
    # Assembling list of averge precision scores for MAP
    avg_prec_scores.append(avg_prec)
    print
    print "ROC AUC SCORE"
    print "Corresponds to area under ROC curve"
    # ROC score
    print metrics.roc_auc_score(true_labels, predicted_labels, average=None)
    print "=" * 80
    print "=" * 80
        
