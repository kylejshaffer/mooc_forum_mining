# Example code for training and testing Perceptron and Naive
# Bayes classifiers using Scikit-Learn and the 20 Newsgroups dataset.

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn import metrics

# This is an optional list from the site example
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'] 
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
shuffle=True, random_state=42, remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories,
shuffle=True, random_state=42, remove=remove)

y_train, y_test = data_train.target, data_test.target
vec = TfidfVectorizer(sublinear_tf=True, min_df=5, stop_words='english')
X_train = vec.fit_transform(data_train.data)
X_test = vec.transform(data_test.data)

def benchmark(clf):
    print "-" * 80
    print "Training: "
    print clf
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    score = metrics.f1_score(y_test, pred)
    print "f1 score: %0.3f" % score
    print "Classification report: "
    print metrics.classification_report(y_test, pred, target_names=categories)
    print "Confusion Matrix: "
    print metrics.confusion_matrix(y_test, pred)
    print()

results = []
print "=" * 80
print "Naive Bayes"
results.append(benchmark(MultinomialNB(alpha=1)))
# results.append(benchmark(BernoulliNB(alpha=1)))
# results.append(benchmark(GaussianNB()))
results.append(benchmark(MultinomialNB(alpha=0.5)))
results.append(benchmark(MultinomialNB(alpha=0.01)))
# results.append(benchmark(BernoulliNB(alpha=0.5)))
# results.append(benchmark(GaussianNB()))
results.append(benchmark(Perceptron(n_iter=50)))