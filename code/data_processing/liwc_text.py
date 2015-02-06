'''
This script extracts only the text from dataset, extracts punctuation, down-cases
the text, and formats data into text file for use with LIWC word-count analysis software.
'''

import string
import pandas as pd
import json, os, string, unicodedata
from bs4 import BeautifulSoup

data_dir = '/Users/kylefth/Documents/mooc_project/Annotations'
os.chdir(data_dir)

features_df = pd.read_csv('/Users/kylefth/Documents/mooc_project/forum_features.csv')
features = features.to_json(orient='records')
features = json.loads(features)

with open('/Users/kylefth/Documents/mooc_project/code_and_threads/forum_comments.json', 'rb') as infile:
    comments = json.load(infile)
    
with open('/Users/kylefth/Documents/mooc_project/code_and_threads/forum_posts.json', 'rb') as infile:
    posts = json.load(infile)

# Need to run this twice - not sure why.
for i in comments:
    if i['deleted'] == 1:
        comments.remove(i)
        
for i in posts:
    if i['deleted'] == 1:
        posts.remove(i)
        
print len(comments) + len(posts)

# Find attributes that uniquely identify each observation
for i in features:
    for j in comments:
        if i['comment'] == 1 and i['id'] == j['id']:
            i['text'] = j['comment_text']
            
for i in features:
    for j in posts:
        if i['comment'] == 0 and i['id'] == j['id']:
            i['text'] = j['post_text']

text_list = [i['text'] for i in features]

exclude = list(set(string.punctuation))
exclude.extend('\n')

clean_text = []
for i in text_list:
    soup = BeautifulSoup(i)
    txt = soup.get_text().lower()
    txt = ''.join(c for c in txt if not c in exclude)
    txt = unicode(txt)
    txt = unicodedata.normalize('NFKD', txt).encode('ascii', 'ignore')
    clean_text.append(txt)

outfile = open('text_lines.txt', 'w')
for i in clean_text:
    outfile.write(i)
    outfile.write('\n')
outfile.close()

initial_lines = open('text_lines.txt', 'rb')
initial_lines = initial_lines.readlines()
# List of cleaned strings
clean_lines = []
for i in initial_lines:
    s = ''
    l = i.split()
    for j in l:
        s += j + ' '
    clean_lines.append(s)

outfile = open('text_lines.txt', 'w')
for i in clean_lines:
    outfile.write(i)
    outfile.write('\n')
outfile.close()
