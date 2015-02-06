# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 16:20:20 2015

@author: kylefth

Python script for running logistic regression classification on MOOC dataset
and performing ablation analysis on features used for classification
"""

# PARAMETER TUNING CODE #
# MAY WANT TO TRY DIFFERENT PARAMETER SPACES #
''' 
# Using Jaime's parameters for tuning
Grid SearchCV for Logistic Regression 1
param_grid = {'C': [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 1]}
# Double-check default settings for LR
clf = LogisticRegression(penalty='l2')
grid_search = GridSearchCV(clf, param_grid)

# Grid SearchCV for Logistic Regression 2
c_range = np.logspace(0, 4, 10)
param_grid = dict(C=c_range)
clf = LogisticRegression(penalty='l2')
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, num_jobs=1)
'''

import os, json, math
import numpy as np
from random import shuffle
from sklearn import preprocessing, metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Change directory to where data is kept and read it in
train_test_pairs = '/Users/kylefth/Documents/mooc_project/mooc_forum_mining/train_test_pairs'

# Feature categories to exclude for ablation analysis
affective = ['ID', 'ASSENT', 'ANXIETY', 'EMOTICONS', 'POSITIVE EMOTION', 'AFFECTIVE PROCESSES', 'ANGER', 'NEGATIVE EMOTION', 'SADNESS']
author = ['ID', 'INSTRUCTOR']
cognitive = ['ID', 'INSIGHT', 'INCLUSION', 'CAUSATION', 'INHIBITION', 'DISCREPANCY', 'TENTATIVENESS', 'COGNITIVE PROCESSES', 'EXCLUSION', 'CERTAINTY']
context = ['ID', 'QUESTION', 'ANSWER', 'ISSUE', 'RESOLUTION', 'POSITIVE', 'NEGATIVE', 'OTHER']
cosine = ['ID', 'PREV', 'AVG', 'MIN', 'MAX', 'STDEV']
current_concerns = ['ID', 'RELIGION', 'LEISURE', 'ACHIEVEMENT', 'HOME', 'WORK', 'MONEY', 'DEATH']
deadline = ['ID', 'DAYS', 'HOURS', 'MINUTES']
linguistic = ['ID', 'TOTAL PRON', 'AUXILIARY VERBS', '1ST SINGULAR', '3RD PLURAL', 'UNIQUE', 'NUMBERS', 'PREPOSITIONS', 'TOTAL 2ND', 'CONJUNCTIONS', 'ARTICLES', 'SWEAR WORDS', 'ABBREVIATIONS','1ST PLURAL','FUNCTION WORDS','3RD SINGULAR','SIXLTR','FUTURE TENSE','PERSONAL PRON','QUANTIFIERS','WPS','PRESENT TENSE','NEGATIONS','IMPERSONAL PRON','ADVERBS','PAST TENSE','COMMON VERBS']
links = ['ID', 'LINKS']
modal = ['ID', 'ABSOLUTE_MOD', 'RELATIVE_MOD']
perceptual = ['ID', 'PERCEPTUAL PROCESSES', 'SEEING', 'HEARING', 'FEELING']
position = ['ID', 'ABSOLUTE_POSIT', 'RELATIVE_POSIT']
post_comment = ['ID', 'POST']
punctuation = ['ID', 'QMARKS', 'PERIOD', 'APOSTRO', 'COLON', 'QUOTE', 'QMARK', 'SEMIC', 'COMMA', 'EXCLAM', 'PARENTH', 'ALLPCT', 'DASH', 'OTHERP']
sentiment = ['ID', 'ABSOLUTE_POS', 'RELATIVE_POS', 'ABSOLUTE_NEG', 'RELATIVE_NEG']
social = ['ID','QMARKS','PERIOD','APOSTRO','COLON','QUOTE','QMARK','SEMIC','COMMA','EXCLAM','PARENTH','ALLPCT','DASH','OTHERP']
spoken = ['ID', 'DIC', 'FILLERS']
answer_unigram = ['ID', 'ACCORD', 'AND', 'BETWEEN', 'BUT', 'CALL', 'CAN', 'CASE', 'CATHI', 'CERTAIN', 'COLLECT', 'CONTAIN', 'CONTENT', 'CONTROL', 'CORE', 'COULD', 'COURS', 'CREATOR', 'DATA', 'DEFIN', 'DEPEND', 'DESCRIB', 'DOCUMENT', 'DTD', 'EASILI', 'ELEMENT', 'ENOUGH', 'ETC', 'EXAMPL', 'FILE', 'FORMAT', 'FRANKENSTEIN', 'GENER', 'HREF', 'HTML', 'HTTP', 'IMAG', 'INTERPRET', 'ISN', 'ITSELF', 'LCNAF', 'LET', 'LINK', 'LITER', 'MAI', 'MEAN', 'META', 'MIGHT', 'MOST', 'NAME', 'NOT', 'OBJECT', 'OFFIC', 'ORG', 'PEOPL', 'POSSIBL', 'PREDIC', 'PROPERTI', 'PURPOS', 'RDF', 'RELATIONSHIP', 'RESOURC', 'SAI', 'SCHEME', 'SEARCH', 'SENS', 'SIMPL', 'SOME', 'SORT', 'SPECIF', 'STANDARD', 'STATEMENT', 'STORE', 'SUCH', 'TAG', 'TARGET', 'TERM', 'THANK', 'THAT', 'THE', 'THEI', 'THEM', 'THERE', 'THINK', 'TITL', 'TRIPL', 'USER', 'USUAL', 'VALU', 'VOCABULARI', 'W3SCHOOL', 'WAI', 'WEEK', 'WHAT', 'WHICH', 'WIKIPEDIA', 'WORD', 'WWW', 'XML', 'YOU', 'YOUNG']
question_unigram = ['ID', 'ADDIT', 'ADVANC', 'ADVIC', 'AGRE', 'ALREADI', 'ANI', 'ANSWER', 'ANYBODI', 'ANYON', 'BETWEEN', 'CAN', 'CATHI', 'CLARIFI', 'COLOUR', 'COME', 'CONSIST', 'COULD', 'CURIOU', 'CURRENT', 'DAI', 'DBPEDIA', 'DECID', 'DEFIN', 'DESCRIB', 'DETAIL', 'DOE', 'EMAIL', 'ENTITI', 'EQUAL', 'EVEN', 'EXCEL', 'EXIST', 'EXPLAIN', 'FACEBOOK', 'FORWARD', 'FREE', 'GLAD', 'GOOD', 'HINT', 'HOPE', 'HOW', 'HTTP', 'INSIGHT', 'INTRODUC', 'KNOW', 'LATER', 'LCNAF', 'LEARN', 'LETTER', 'LIST', 'META', 'MICROFORMAT', 'MISTAK', 'MUCH', 'NOTE', 'ONTOLOG', 'ORDER', 'PAINT', 'PERIOD', 'PLEAS', 'PSEUDONYM', 'QUESTION', 'REASON', 'RECIP', 'RECORD', 'REGARD', 'RELAT', 'REPEAT', 'SCHEMA', 'SCHEME', 'SENTENC', 'SIGNATUR', 'SOLUT', 'SOMEON', 'STUCK', 'TAKE', 'THAT', 'THE', 'THERE', 'THESAURU', 'THOUGHT', 'TOO', 'TRACK', 'TRI', 'TROUBL', 'TWAIN', 'VALU', 'VERI', 'VIDEO', 'WAREHOUS', 'WEBSIT', 'WELL', 'WHAT', 'WHERE', 'WHY', 'WONDER', 'WOULD', 'WOULDN', 'WRONG', 'WROTE']
issue_unigram = ['ID', '2ND', 'ADDIT', 'AGAIN', 'ALT', 'AMAZONAW', 'ANSWER', 'ASSIGN', 'ATTEMPT', 'BECAUS', 'BOTHER', 'BRACKET', 'BROWSER', 'BUT', 'BUTTON', 'CANNOT', 'CHARACT', 'CHROME', 'CONFUS', 'CONTINU', 'CORRECT', 'COUNT', 'COURSERA', 'DEADLIN', 'DETAIL', 'DUE', 'ERROR', 'EXPLAN', 'EXTEND', 'FACTUAL', 'FAIL', 'FINAL', 'FIREFOX', 'FRUSTRAT', 'FULL', 'GET', 'GIVE', 'GOT', 'GRADE', 'HOMEWORK', 'HREF', 'HUGE', 'IMG', 'INCORRECT', 'INTEREST', 'ISSU', 'LINE', 'LOAD', 'LOWER', 'MAC', 'MARK', 'MESSAG', 'METADATA', 'MISS', 'MISTAK', 'NOT', 'NOTIC', 'OCCUR', 'OMIT', 'OPTION', 'OTHER', 'PAGE', 'PLAYER', 'PLEAS', 'PROBLEM', 'QUESTION', 'QUIZ', 'QUIZZ', 'QUOT', 'REACH', 'REPLAC', 'REQUEST', 'REST', 'RESULT', 'SAW', 'SCORE', 'SCREENSHOT', 'SELECT', 'SHOWN', 'SHUTDOWN', 'STAFF', 'STRONG', 'SUBMISS', 'SUBMIT', 'TARGET', 'TEST', 'THE', 'THERE', 'THI', 'TRI', 'TRULI', 'TYPE', 'TYPO', 'VIDEO', 'WEEK', 'WEEKLI', 'WERE', 'WHEN', 'WINDOW', 'WRONG', 'YOUR']
issue_res_unigram = ['ID', '001', 'ABOUT', 'AGAIN', 'AGRE', 'ALL', 'AMBIGU', 'AND', 'APOLOG', 'APPAR', 'ATTENT', 'AVAIL', 'AVOID', 'BEEN', 'BLANK', 'BOARD', 'BRING', 'CALENDAR', 'CAN', 'CATCH', 'CAUS', 'CHECK', 'CLASS', 'CONFIRM', 'CONFUS', 'CONTENT', 'COURS', 'COURSERA', 'CREDIT', 'DATA', 'DECIS', 'DESCRIB', 'DETERMIN', 'DOE', 'DOWNLOAD', 'EACH', 'ENCOUNT', 'EUROPEANA', 'EXAMPL', 'EXTRA', 'FACE', 'FEEDBACK', 'FIX', 'FORUM', 'FROM', 'FURTHER', 'GET', 'GOAL', 'GRANT', 'GREAT', 'HOMEWORK', 'INFORM', 'INSTEAD', 'INTEREST', 'ISSU', 'LIKE', 'MEAN', 'MUCH', 'NOT', 'NOW', 'ORG', 'OUR', 'OUT', 'PAGE', 'PLAYER', 'POINT', 'PORTAL', 'PROBLEM', 'QUIZ', 'REACH', 'REALLI', 'RECEIV', 'REMOV', 'REPORT', 'RESOLV', 'SCORE', 'SCREENSHOT', 'SELECT', 'SET', 'SINCER', 'SLIDE', 'SLIGHTLI', 'SOME', 'SOON', 'SORRI', 'START', 'STUDENT', 'SUBJECT', 'SUBMISS', 'SUBMIT', 'THANK', 'THIRD', 'THREAD', 'UNDER', 'UNIT', 'UPDAT', 'VERI', 'WHERE', 'WILL', 'WITH', 'WOULD']
positive_ack_unigram = ['ID', 'ADDIT', 'AGRE', 'AMAZ', 'ANI', 'ANSWER', 'ANYON', 'APPLIC', 'APPRECI', 'ASSIGN', 'AWESOM', 'BETWEEN', 'BLANK', 'BROWSER', 'BUT', 'CALL', 'CAN', 'CATHI', 'CHROME', 'COM', 'CONTAIN', 'CONTENT', 'CORRECT', 'COURS', 'COURSERA', 'CREAT', 'DATE', 'DEPEND', 'DESCRIB', 'DETAIL', 'DOE', 'DON', 'ELEMENT', 'ENJOI', 'ERROR', 'EXCEL', 'FANTAST', 'FORUM', 'FROM', 'GLAD', 'GOOD', 'GREAT', 'HREF', 'HTML', 'HTTP', 'INCLUD', 'INDEX', 'ISSU', 'JUST', 'LEARN', 'LINK', 'MARK', 'MEAN', 'META', 'METADATA', 'MUCH', 'NAME', 'NOT', 'NOTE', 'NUMBER', 'OMIT', 'ONLI', 'OPTION', 'ORG', 'OTHER', 'PAGE', 'PARA', 'POMERANTZ', 'PROBLEM', 'PROF', 'PROPERTI', 'QUE', 'QUESTION', 'RECORD', 'SCHEME', 'SHOULD', 'SINC', 'SORRI', 'STRONG', 'SUBMIT', 'TARGET', 'TEAM', 'TERM', 'THANK', 'THAT', 'THE', 'THERE', 'THREAD', 'TITL', 'TOO', 'TYPE', 'TYPO', 'VALU', 'VERI', 'VIDEO', 'WHAT', 'WHICH', 'WINDOW', 'WONDER', 'WOW', 'WWW']
negative_ack_unigram = ['ID', 'ACCEPT', 'ALTERN', 'ANONYM', 'ANSWER', 'ANYWAI', 'ARMIN', 'ASSIGN', 'ASSUM', 'AWAI', 'BASE', 'BELIEV', 'BEYOND', 'BLUE', 'CALL', 'CHALLENG', 'CLEARLI', 'CODE', 'CONSIDER', 'COOL', 'CREATION', 'CREATOR', 'DATE', 'DIDN', 'DIRECT', 'DIRECTLI', 'DISAGRE', 'DISAPPOINT', 'DISCIPLIN', 'EFFORT', 'ENTER', 'ENTRI', 'EXACTLI', 'EXAMPL', 'EXERCIS', 'EXPECT', 'EXPRESS', 'FAIL', 'FEEDBACK', 'FLEXIBL', 'FORM', 'FRUSTRAT', 'GET', 'HATE', 'IMPROV', 'INSTRUCTOR', 'INTENT', 'LESS', 'LIMIT', 'MAI', 'MAIN', 'MANI', 'MARK', 'METADATA', 'NECESSARILI', 'NEG', 'NOT', 'OBJECT', 'ONC', 'ONLI', 'OPINION', 'OUT', 'PERHAP', 'PERSON', 'PLATFORM', 'POINT', 'POSIT', 'PRE', 'PREDIC', 'PROBABL', 'PROPERTI', 'PROVID', 'QUESTION', 'RANG', 'REAL', 'REASON', 'RESPONS', 'SEMANT', 'SERIOUS', 'SHOULD', 'SIMPLI', 'SKY', 'SOMETH', 'SPACE', 'STATEMENT', 'STUFF', 'SUBMIT', 'SUPPOS', 'TEACH', 'TEXT', 'THANK', 'THI', 'THREAD', 'TURN', 'UNDERSTOOD', 'W3SCHOOL', 'WHAT', 'WHERE', 'WHITE', 'WHY', 'WIDE']
other_unigram = ['ID', '2013', 'ACTIV', 'ADD', 'AGAIN', 'AGO', 'AGRE', 'ANALYT', 'ANSWER', 'BAR', 'BOTTOM', 'BOX', 'BUT', 'BUTTON', 'CATALOG', 'CLICK', 'COM', 'COMMENT', 'COMO', 'CON', 'CONSULT', 'CONVERS', 'CORRECT', 'CORRESPOND', 'CURRENT', 'CURSO', 'DEL', 'DISCUSS', 'DROPBOX', 'ENHANC', 'EST', 'EVERYBODI', 'EVERYON', 'EXAMPL', 'EXPAND', 'EXPLOR', 'FOR', 'FORUM', 'FROM', 'GRACIA', 'GREEN', 'GREI', 'GROUP', 'HAVE', 'HEI', 'HELLO', 'HREF', 'HTTP', 'ICON', 'INDIA', 'ISLAND', 'JOIN', 'LINK', 'LINKEDIN', 'LIVE', 'METADATA', 'MLI', 'MUCHA', 'NAVIG', 'NOT', 'NUMBER', 'OBJECT', 'PARA', 'PARTICIP', 'PERO', 'PHD', 'PHP', 'POR', 'PROFIT', 'QUE', 'QUESTION', 'REPUT', 'RESPOND', 'SERVIC', 'SIDE', 'SKILL', 'SOMETH', 'STAT', 'STREAM', 'STUDENT', 'TARGET', 'THANK', 'THAT', 'THE', 'THERE', 'THINK', 'THREAD', 'TITL', 'TODO', 'TOTAL', 'TOWARD', 'UNIQU', 'UNIVERS', 'VOTE', 'WATCH', 'WHAT', 'WORD', 'WOULD', 'WRONG', 'WWW', 'YOU']

# Function for reading in train-test pairs
def train_test(label):
    os.chdir(train_test_pairs + '/' + str(label) + '/pairs')
    files = os.listdir(os.getcwd())

    pairs = []
    for i in files:
        if i[-5:] == '.json':
            with open(i, 'rb') as infile:
                data = json.load(infile)
                pairs.append(data)
    return pairs

# Function for removing set of features for ablation analysis
def run_ablation(features, data):
    for d in data:
        for k in d.keys():
            if k in features:
                del d[k]

# Function to run cross validation
def run_cross_val(label, features):
    pairs = train_test(label)
    avg_prec_scores = []
    for p in pairs:
        current_data = p
        # Training set
        X = current_data[0]; # print len(X[0].keys())
        # Test set
        y = current_data[1]; # print len(y[0].keys())
        
        # Removing feature set for ablation analysis
        run_ablation(features, X) # print len(X[0].keys())
        run_ablation(features, y) # print len(y[0].keys())
        
        train_labs = [int(i['LABEL']) for i in current_data[0]]
        test_labs = [int(i['LABEL']) for i in current_data[1]]
        
        # Evening out training data for 50/50 split
        pos_class = [i for i in X if i['LABEL'] == 1]
        neg_class = [i for i in X if i['LABEL'] == 0]
        
        # Deal with ratios as floats at first then take floor or ceiling division
        # Run this and look at ratio calculations and make sure these are right
        ''' New code here should account for class lists with length of zero'''
        if (len(neg_class) > len(pos_class)) and len(pos_class) > 0:
            ratio = int(math.ceil(float(len(neg_class)) / float(len(pos_class))))
            # print "Original class ratio is %s: 1, negative_class: positive_class" % ratio
            pos_class = pos_class * ratio
            # print "The positive class now contains %s instances" % len(pos_class)

        elif (len(pos_class) > len(neg_class)) and len(pos_class) > 0:
            ratio = int(math.ceil(float(len(pos_class)) / float(len(neg_class))))
            # print "Original class ratio is %s: 1, positive_class: negative_class" % ratio
            neg_class = neg_class * ratio
            # print "The negative class now contains %s instances" % len(neg_class)
            
        else: pass

        X = pos_class + neg_class
                
        train_labels = [i['LABEL'] for i in X]
        test_labels = [i['LABEL'] for i in y]
        
        # Converting labels to NP arrays for use in Sklearn classifiers
        vec_train_labels = np.array(train_labels)
        vec_test_labels = np.array(test_labels)
        
        # Remove target labels from data instances
        X_nolab = X
        for i in X_nolab:
            if 'LABEL' in i.keys():
                del i['LABEL']
        
        y_nolab = y
        for i in y_nolab:
            del i['LABEL']
        
        # Initialize vectorizer object to convert data instances to 
        # numpy format for sklearn
        vec = DictVectorizer()
        vec_train = vec.fit_transform(X_nolab).toarray()
        vec_test = vec.fit_transform(y_nolab).toarray()
        
        # Min-max normalization
        scaler = preprocessing.MinMaxScaler()
        scaled_train_data = scaler.fit_transform(vec_train)
        scaled_test_data = scaler.transform(vec_test)
            
        # Initialize chosen classifier instance, train it on 
        # data, and generate predictions based on this model
        clf = LogisticRegression(penalty='l2')
        # Set parameters for grid search algorithm to tune C paramter
        param_grid = {'C': [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 1]}
        # Initialize grid search algorithm and return model with tuned parameters
        grid_search = GridSearchCV(clf, param_grid=param_grid, scoring="average_precision", cv=5)
        # Fit the model with grid-search-tuned parameters
        grid_search.fit(scaled_train_data, vec_train_labels)
        ''' Code below may be necessary for PR-curve plots with matplotlib'''
        '''y_score = clf.fit(scaled_train_data, vec_train_labels).decision_function(scaled_test_data)'''
        # Predicted labels in test set
        predicted = grid_search.predict(scaled_test_data)
        
        # Print out metrics for evaluation
        print_metrics(vec_test_labels, predicted, avg_prec_scores)
    
    # print
    mean_avg_prec = float(sum(avg_prec_scores)) / float(len(avg_prec_scores))
    print "MAP: %s" % mean_avg_prec

# Function for printing metrics
def print_metrics(true_labels, predicted_labels, avg_prec_scores):
    avg_prec_scores = avg_prec_scores
    # Averge precision
    avg_prec = metrics.average_precision_score(true_labels, predicted_labels, average=None)
    # Assembling list of averge precision scores for MAP
    avg_prec_scores.append(avg_prec)
        
if __name__ == "__main__":
    # Run the program
    label_list = ['question', 'answer', 'issue', 'issue_resolution', 'positive_ack', 'negative_ack', 'other']
    feature_group_list = [affective, author, cognitive, context, cosine, current_concerns, deadline, linguistic, links, modal, perceptual, position, post_comment, punctuation, sentiment, social, spoken]
    feature_group_names = ['affective', 'author', 'cognitive', 'context', 'cosine', 'current_concerns', 'deadline', 'linguistic', 'links', 'modal', 'percpetual', 'position', 'post_comment', 'punctuation', 'sentiment', 'social', 'spoken']
    # Call cross_val and ablation study functions to actually run the analysis
    for lab in label_list:
        print "=" * 30 + lab.upper() + "=" * 30
        print "=" * 30 + "NONE" + "=" * 30
        run_cross_val(lab, [])
        for n, f in enumerate(feature_group_list):
            print "=" * 30 + feature_group_names[n].upper() + "=" * 30
            run_cross_val(lab, f)
        print "=" * 30 + "UNIGRAM" + "=" * 30
        if lab == 'question': run_cross_val(lab, question_unigram)
        elif lab == 'answer': run_cross_val(lab, answer_unigram)
        elif lab == 'issue': run_cross_val(lab, issue_unigram)
        elif lab == 'issue_resolution': run_cross_val(lab, issue_res_unigram)
        elif lab == 'positive_ack': run_cross_val(lab, positive_ack_unigram)
        elif lab == 'negative_ack': run_cross_val(lab, negative_ack_unigram)
        elif lab == 'other': run_cross_val(lab, other_unigram)
        print