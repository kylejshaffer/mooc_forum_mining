########## LIST OF FEATURES ##########
# modals = modal verbs present in post
# link = hyperlink present in post
# question_begin = first sentence of post contains a question word
# question_total = question word occurs anywhere in post
# comment = binary indicating whether post is a comment or not
# votes = up-votes or down-votes of post from other classmates
# tag_tokens = html tags present in raw post
# word_tokens = downcased words present in post or comment
# post_length = number of word tokens in post
# pos_words = number of word tokens in positive sentiment wordlist
# neg_words = number of word tokens in negative sentiment wordlist
# is_instructor = binary flag as to whether post was written by instructor or TA (1 yes, 0 no)
# pos_emoticons = number of positive emoticons in thread
# neg_emoticons = number of negative emoticons in thread
# pos_tags = syntactic part-of-speech tags
# prev_cosine_score = cosine similarity score between the post under consideration and previous post in thread

# TO-DO: (1) merge in labels for posts, (2) figure out how to deal with redundant labels, 
# (3) figure out whether we're working with deleted posts or not

import os, json, re
import unicodedata
import pandas as pd
from itertools import islice, chain
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize, pos_tag
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

data_dir = '/Users/kylefth/Documents/mooc_project/code_and_threads'
scripts_dir = '/Users/kylefth/Documents/mooc_project'
os.chdir(data_dir)

with open('threads.json', 'rb') as infile:
    threads = json.load(infile)

os.chdir(scripts_dir)

# Adding positive and negative sentiment features
with open('negative.txt', 'rb') as infile:
    neg_words = unicode(infile.read(), errors='ignore')
    neg_words = neg_words.split('\n')
    neg_words = neg_words[35:-1]
    
with open('positive.txt', 'r') as infile:
    pos_words = unicode(infile.read(), errors='ignore')
    pos_words = pos_words.split('\n')
    pos_words = pos_words[35:-1]

# Incorporating positive and negative words as features
for i in threads:
    for j in i['thread_content']:
        pos_count = [w for w in j['word_tokens'] if w in pos_words]
        neg_count = [w for w in j['word_tokens'] if w in neg_words]
        j['pos_words'] = len(pos_count)
        j['neg_words'] = len(neg_count)


# Tag list for comparing and extracting tags from text
tags = ['<b>','<p>','<i>','<br/>','<br />','<ul>','<li>','<code>','<div>','<em>',
'<footer>','<h1>','<h2>','<h3>','<h4>','<h5>','<h6>','<head>','<iframe>','<img>','<ol>',
'<script>','<span>','<strong>','<table>','<tbody>','<td>','<tr>','<th>','<title>','<u>','<a>']

# Generating counts of HTML tags as features
for i in threads:
    for j in i['thread_content']:
        html_set = set(j['tag_tokens'])
        for k in html_set:
            j[k] = j['tag_tokens'].count(k)

# List of modal verbs
modals = ['can', "can't", 'could', "couldn't",'should', "shouldn't", 'may', 
'might', 'must', 'ought', 'shall', 'will', "won't", 'would', "wouldn't"]

# Incorporating modal verbs as features
for i in threads:
    for j in i['thread_content']:
        mod_verbs = [w for w in j['word_tokens'] if w in modals]
        j['modals'] = len(mod_verbs)

# Finding links
link_finder = re.compile('<a href=(.*)</a>')
target_finder = re.compile('<a target=(.*)</a>')

# Incorporating links as features
for i in threads:
    for j in i['thread_content']:
        if 'post_text' in j:
            links = re.findall(link_finder, j['post_text'])
            targets = re.findall(target_finder, j['post_text'])
            if len(links) > 0 or len(targets) > 0:
                j['link'] = 1
            else: j['link'] = 0
        elif 'comment_text' in j:
            links = re.findall(link_finder, j['comment_text'])
            targets = re.findall(target_finder, j['comment_text'])
            if len(links) > 0 or len(targets) > 0:
                j['link'] = 1
            else: j['link'] = 0

# List of question words
quest_words = ['who', 'what', 'where', 'when', 'why', 'how']

# Finding question words that occur in first sentence of post
for i in threads:
    for j in i['thread_content']:
        if 'post_text' in j:
            try:
                t = BeautifulSoup(j['post_text']).get_text().lower()
                w = word_tokenize(sent_tokenize(t)[0])
                q = [l for l in w if l in quest_words]
                if len(q) > 0:
                    j['question_begin'] = 1
                else: j['question_begin'] = 0
            except:
                j['question_begin'] = 0
        elif 'comment_text' in j:
            t = BeautifulSoup(j['comment_text']).get_text().lower()
            w = word_tokenize(sent_tokenize(t)[0])
            q = [l for l in w if l in quest_words]
            if len(q) > 0:
                j['question_begin'] = 1
            else: j['question_begin'] = 0

# Question words that occur anywhere in post
for i in threads:
    for j in i['thread_content']:
        if 'post_text' in j:
            text = BeautifulSoup(j['post_text']).get_text().lower()
            w = word_tokenize(text)
            q = [l for l in w if l in quest_words]
            j['question_total'] = len(q)
        elif 'comment_text' in j:
            text = BeautifulSoup(j['comment_text']).get_text().lower()
            w = word_tokenize(text)
            q = [l for l in w if l in quest_words]
            j['question_total'] = len(q)
            
# Adding flag feature indicating whether post is by instructor or not
instructor_id = '4911e06d46c322ab6dca1d86bc43283e45c1eb72'
ta_id = '67404b48e5647e076b49a4fd212492e6f58c3a63'
for i in threads:
    for j in i['thread_content']:
        if j['forum_user_id'] == instructor_id or j['forum_user_id'] == ta_id:
            j['is_instructor'] = 1
        else: j['is_instructor'] = 0        

# Lists of positive and negative emoticons
pos_emoticons = [':)', ': )', ':-)', ';)', ';-)',
':D', ':-D', '8-D', '8D', 'X-D', 'xD', ':]', '=)', ':^)', ':P', ':-P', ';P',
';-P', 'xP', 'x-P', 'XP', 'X-P', '=P', ':->', ':3', ':-}', '(^.^)', 
'^^', '>.<', 'o_O', 'O.O', ':$']
neg_emoticons = [':(', ':-(', ':[', ':-[', ':{', ':-{', ':|', ':-|', '>:[',
':c', ':-c', ':-<', ':<', ':=||', '>:(', ':@', ":'-(", ":'(", ':L', ':S']

# Adding positive and negative emoticons as features
for i in threads:
    for j in i['thread_content']:
        if 'post_text' in j:
            soup = BeautifulSoup(j['post_text'])
            txt = soup.get_text().split()
            pos = [w for w in txt if w in pos_emoticons]
            neg = [w for w in txt if w in neg_emoticons]
            j['pos_emoticons'] = len(pos)
            j['neg_emoticons'] = len(neg)
        if 'comment_text' in j:
            soup = BeautifulSoup(j['comment_text'])
            txt = soup.get_text().split()
            pos = [w for w in txt if w in pos_emoticons]
            neg = [w for w in txt if w in neg_emoticons]
            j['pos_emoticons'] = len(pos)
            j['neg_emoticons'] = len(neg)

# Adding POS-tags
for i in threads:
    for j in i['thread_content']:
        if 'post_text' in j:
            soup = BeautifulSoup(j['post_text'])
            txt = soup.get_text().split()
            j['pos_tags'] = pos_tag(txt)
        if 'comment_text' in j:
            soup = BeautifulSoup(j['comment_text'])
            txt = soup.get_text().split()
            j['pos_tags'] = pos_tag(txt)

# Generating counts for each POS tag as a feature
for i in threads:
    for j in i['thread_content']:
        pos = [i[1] for i in j['pos_tags']]
        set_pos = set(pos)
        for k in set_pos:
            j[k] = pos.count(k)
        del j['pos_tags']
        
# Adding post index as features
for i in threads:
    for j,k in enumerate(i['thread_content']):
        if len(k) > 1:
            if j == 0:
                k['first_index'] = 1
            elif j == 1:
                k['second_index'] = 1
        else:
            k['first_index'] = 1

for i in threads:
    if len(i['thread_content']) > 2:
        i['thread_content'][-1]['last_index'] = 1

# Extracting list of post/comment strings for transformation to 
# TFIDF vectors.
thread_list = []

for i in threads:
    for j in i['thread_content']:
        if 'post_text' in j:
            soup = BeautifulSoup(j['post_text'])
            thread_list.append(soup.get_text().lower())
        elif 'comment_text' in j:
            soup = BeautifulSoup(j['comment_text'])
            thread_list.append(soup.get_text().lower())

tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform(thread_list)

# Using TF-IDF vectors as temporary features for cosine similarity scores
count = 0
for i in threads:
    for j in i['thread_content']:
        row = matrix.getrow(count)
        j['tfidf'] = row
        count += 1
        
# Create sliding window function to do pairwise comparisons for 
# cosine similarity function
def sliding_window(a):
    z = (islice(a, i, None) for i in range(2))
    return zip(*z)

# Iterate through threads, create ordered list of TF-IDF weights per thread,
# generate pairwise cosine similarity scores for each post in thread

post_list = []
for i in threads:
    post_list.append(i['thread_content'])
    # tfidf_weights = [j['tfidf'] for j in i['thread_content']]
    
# Option 1: Flatten out each list and figure out a "skipping method"
# so cosine scores are not assigned to first post of each thread
# Option 2: Iterate through list of posts, compute cosine similarity
# for each pair of posts, add this score as attribute to each post
    
for i in post_list:
    for j, k in enumerate(sliding_window(i)):
        cos = cosine_similarity(k[0]['tfidf'], k[1]['tfidf'])[0][0]
        i[j+1]['prev_cosine_score'] = cos
        
for i in post_list:
    try:
        i[0]['prev_cosine_score'] = 0.
    except:
        print post_list.index(i)
        break

# Flatten out nested list of forum posts
flat_posts = list(chain.from_iterable(post_list))

df = pd.DataFrame(flat_posts)

delete_list = ['tag_tokens', 'stickied', 'post_text', 'word_tokens', 'text_type', 'edit_time', 'anonymous', 'approved', 'is_spam', 'original', \
'#', '$', "''", ",", ".", ":", "``", 'comment_text', 'post_time', 'tfidf', 'user_agent']

for i in delete_list:
    if i in df.columns:
        df.pop(i)

df.to_csv('forum_features.tsv', sep="\t")

for i in df.columns:
    df[i].fillna(0, inplace=True)

df.to_csv('forum_features_noNAs.tsv', sep="\t")
