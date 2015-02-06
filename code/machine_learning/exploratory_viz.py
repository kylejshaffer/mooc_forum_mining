import os
import seaborn as sns
import matplotlib as plt
import pandas as pd

data_dir = '/Users/kylefth/Documents/mooc_project/Annotations'
os.chdir(data_dir)

df = pd.read_csv('all_labels_consolidated.csv')
df2 = df.reindex(columns=['post_length', 'prev_cosine_score', 'pos_words', 'neg_words', 'modals', 'label_1', 'is_instructor', 'comment'])
print df2.shape
df3 = df.reindex(columns=['post_length', 'modals', 'label_1'])
print df3.shape

# Pairwise matrix of variables
sns.pairplot(df2, 'label_1', size=3)

# Pairwise matrix 2
sns.pairplot(df3, 'label_1', size=3)