"""

# -*- coding: utf-8 -*-
Created on          18 Jun 2019 at 4:11 PM
Author:             Arvind Sachidev Chilkoor  
Created using:      PyCharm
Name of Project:    NLTK based SMS SPAM Filter

Description:  
            The script attempts to the nltk library along with Sci-Kit Learn to detect a set of data containing SMS
            messages, as either HAM or SPAM. The script uses various ML classifiers to predict the outcome.
            It also makes use VotingClassifier from sklearn.ensemble and wraps them in nltk to combine the models and
            produce more accurate predictions/result

            Dataset has been borrowed from for a public domain repository with MIT License

            At the end a confusion matrix is printed to show as to how many where classified as HAM or SPAM
"""

import nltk
import sys
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Loading the dataset of SMS
df = pd.read_table('SMSSpamCollection', header= None, encoding='utf-8')

# print useful information about the dataset
print("\n")
print(df.info())
print(df.head())

# Checking for class distributions
classes = df[0]
print(classes.value_counts()) # This returns object containing counts of unique values.

# PRE-PROCESSING THE DATA
# Convert the class labels to binary values, wherein 0 = ham and 1 = spam messages
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

# Test Print first 10 lines
print("\n")
print(classes[:10]) # First 10 lines
print(Y[:10]) # Their binary equivalent

# To store the SMS data
text_messages = df[1]

# Test Print
print("\n")
print(text_messages)

# Using REGULAR EXPRESSIONS to replace email addresses, URLS/web addresses, phone numbers, other numbers
# Reference taken from regexlib.com

# To replace email addresses with 'EmailID'
processed = text_messages.str.replace(r'^([0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*@([0-9a-zA-Z][-\w]*[0-9a-zA-Z]\.)'
                                      r'+[a-zA-Z]{2,})$','EmailID')


# To replace web addresses with 'WebAddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','WebAddress')

# To replace the currency symbols such as $ and £ with 'MoneySymbols'
processed = processed.str.replace(r'£|\$', 'MoneySymbols')

# To replace 10 digit phone numbers including paranthesis, spaces, no spaces, dashes with 'PhoneNumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'PhoneNumber')

# To replace general numbers with 'Number'
processed = processed.str.replace(r'\d+(\.\d+)?','Number')

# To remove punctuation from the messages
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# To remove the whitespace between terms with a single space
processed = processed.str.replace(r'\s+',' ')

# To remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')


# To convert all the words to lowercase, inorder that all words are uniformly captured.
processed = processed.str.lower()

# Test print
print("\n")
print(processed)

# Stopwords are those words that do not add any meaning to the sentences, neither do they change the meaning of the
# sentence
StopWords = set(stopwords.words('english'))

# To preprocess the list of SMS by filtering the Stop Words
# using a Lambda function join the other words in an empty string
# Here the final string will contain all the words apart from the Stop Words
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in StopWords))

# To reduce the words to the base/root stem using Porter Stemmer, part of the NLTK library
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

# To create features for machine learning algorithms, using the words in each text message.
# Tokenizing each word from the set for the most common words. These words will be the features that machine learning
# algorithm will be using.


# To create a bag of words, that will form the basis of the features for ML Algorithm

AllWords = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        AllWords.append(w)

AllWords = nltk.FreqDist(AllWords)


# Test print the total number of words and the 15 most common words
print('\nNumber of words: {}'.format(len(AllWords)))
print('\nMost common words: {}'.format(AllWords.most_common(20)))

# To create the 1500 of the most common words as features
word_features = list(AllWords.keys())[:2000]

# To create a function that will determine which of the 1500 word features are in the review of the messages.

def find_words(message):
    words = word_tokenize(message)    # Tokenizing the message
    features = {}        # Empty Dictionary
    for word in word_features:
        features[word] = (word in words)

    return features


# Test Print for the first set of messages in processed

features = find_words(processed[0])
print("\n-----------------")
for key, value in features.items():
    if value == True:
        print(key)

# To find the features in all the messages
# To Zip all processed messages in Y (i.e.) class labels
messages = list(zip(processed, Y))

# To create seed starting point.
seed = 1
np.random.seed = seed

# To shuffle the messages to create greater randomness.
np.random.shuffle(messages)


# To call the find_words function for each of the SMS. Text and Label are contained messages since it has been zipped
word_sets = [(find_words(text), label) for (text, label) in messages]


# To split the data into Training Set and Testing Set
training, testing = model_selection.train_test_split(word_sets, test_size=0.2, random_state=seed)


# Test Print total training set data and test set data i.e. numbers
print("\n\n")
print('No. of messages in Training Set: ', format(len(training)))
print('No. of messages in Testing Set: ',format(len(testing)))


# To create the ML Classifiers
# To define the models to train

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "Stochastic Gradient Descent Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
    ]

models = zip(names, classifiers)

# To wrap the models within NLTK
# Importing SKlearnClassifier
from nltk.classify.scikitlearn import SklearnClassifier

"""         
# To parse through the various ML classifiers to see which offers the highest accuracy of classifying is a SMS is
# HAM or SPAM
print("\n----------------------------------------------------------")
for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy:    {}".format(name, accuracy))
"""

# To get a higher accuracy, using the sklearn.ensemble.VotingClassifier, which combines all the above classifier to
# qualify if it is a HAM or SPAM, since there a total of 7 classifiers used, the ratio 4:3 will be the deciding factor

from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

# Same as previous
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

# Same as previous
models = list(zip(names, classifiers))

# VotingClassifier (Parameters: estimators = models; voting = 'hard' since classifiers not optimized; n_jobs = 1
# to use single core of CPU, -1 will result in using all the cores of the CPU, to NOTE: -1 can lead to overflow

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = 1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_ensemble, testing)*100
print("\n***************************************************************")
print("\nEnsemble Method Voting Classifier: Accuracy: {}".format(accuracy))



# To create a Class Label Prediction for the Testing Set
txt_features, labels = list(zip(*testing))

prediction = nltk_ensemble.classify_many(txt_features)


# To print a confusion matrix and a classification report
print("\n")
print(classification_report(labels, prediction))


# To create a Pandas Dataframe for confusion matrix, to indicate is how much was classified as HAM or SPAM
i = pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])

# To print the confusion matrix
print("\n\n")
print(i)