import pandas as pd
import json, os

data_dir = '/Users/kylefth/Documents/mooc_project'
data_dir2 = '/Users/kylefth/Documents/mooc_project/Annotations'
# Load in features
os.chdir(data_dir)
features = pd.read_csv('mooc_features.csv')
# Load in majority vote labels
os.chdir(data_dir2)
maj_vote = pd.read_csv('majority_votes.csv')
# Split by whether instance is a comment or post
df_comm = features.ix[features.comment == 1]
df_post = features.ix[features.comment == 0]

comm_json = df_comm.to_json(orient = 'records')
post_json = df_post.to_json(orient = 'records')
comm_json = json.loads(comm_json)
post_json = json.loads(post_json)

maj_comment = maj_vote.ix[maj_vote.type == 'comment']
maj_post = maj_vote.ix[maj_vote.type == 'post']

maj_comm_json = maj_comment.to_json(orient = 'records')
maj_post_json = maj_post.to_json(orient = 'records')
maj_comm_json = json.loads(maj_comm_json)
maj_post_json = json.loads(maj_post_json)
# Merge in labels
for i in maj_post_json:
    for j in post_json:
	if i['id'] == j['id']:
		j['label_1'] = i['label_1']
		j['label_2'] = i['label_2']
		j['label_3'] = i['label_3']

for i in maj_comm_json:
    for j in comm_json:
	if i['id'] == j['id']:
		j['label_1'] = i['label_1']
		j['label_2'] = i['label_2']
		j['label_3'] = i['label_3']

df_comm = pd.DataFrame(comm_json)
df_post = pd.DataFrame(post_json)
all_labels = df_comm.append(df_post)
# Single out instances that need to be duplicated/have more than one label
all_labels_json = all_labels.to_json(orient = 'records')
all_labels_json = json.loads(all_labels_json)
reps = [i for i in all_labels_json if not i['label_2'] == None]
reps2 = [i for i in all_labels_json if not i['label_3'] == None]

for i,j in enumerate(reps):
    if j['label_3'] != None:
	reps.pop(i)

reps_dup = reps
reps_dup2 = reps2

for i in reps_dup:
    i['label_1'] = i['label_2']

for i in reps_dup2:
	i['label_1'] = i['label_3']

for i in reps_dup:
    try:
	del i['label_2']
	del i['label_3']
    except:
        pass

for i in reps_dup2:
    try:
	del i['label_2']
	del i['label_3']
    except:
        pass

for i in all_labels_json:
    try:
	del i['label_2']
	del i['label_3']
    except: pass

all_labels_json = all_labels_json + reps_dup
all_labels_json = all_labels_json + reps_dup2

all_labels = pd.DataFrame(all_labels_json)
