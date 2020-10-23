#X-TRAIN - TRAIN.txt questions
#X-TEST  - DEV-questions.txt 
#y-TRAIN - TRAIN.txt labels
#y-TEST - DEV-labels.txt 
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import sys

#print ('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))

#open files
fStop = open("stopwords.txt","r") #custom stopwords
fTrain=open(sys.argv[2], "r")
fQuestions=open(sys.argv[3], "r") #test file
fLabels=open("training/DEV-labels.txt", "r") #test file

stopwords = []
for word in fStop.readlines():
    stopwords.append(word.replace("\n",""))
#print(stopwords)

#separating questions in test set
testLines = fQuestions.readlines()
devQuestions = []
for l in testLines:
    for word in stopwords:
        l = l.replace(" "+ word + " ", " ")
        l = l.replace("What's","what is")
    devQuestions.append(l)
#print(devQuestions)
testLines = fLabels.readlines()
devLabels = []
for l in testLines:
    devLabels.append(l.replace('\n',''))

#separating questions from targets
lines = fTrain.readlines()
labels = []
questions = []

for l in lines:
    lSplit = l.split(' ',1)
    labels.append(lSplit[0])
    for word in stopwords:
        lSplit[1] = lSplit[1].replace(" "+ word + " ", " ")
        lSplit[1] = lSplit[1].replace("What's","what is")
    questions.append(lSplit[1])
fTrain.close()
fQuestions.close()
fLabels.close()

import numpy 
questions = numpy.array(questions)
devQuestions = numpy.array(devQuestions)
#print("labels:", len(labels))
#print("questions:", len(questions))

#preprocess text
import re
import string
from nltk.stem import WordNetLemmatizer 

lemmatizer=WordNetLemmatizer()
translator = str.maketrans('', '', string.punctuation)
porter = PorterStemmer()

for index, question in enumerate(questions):
    questions[index] = question.lower() #lowercase
    questions[index] = re.sub(r'\d+', '', questions[index])  #remove numbers
    questions[index] = questions[index].translate(translator)
    finalstr = ""
    splitted = questions[index].split(' ')
    for word in splitted:
        finalstr = finalstr + " " +porter.stem(lemmatizer.lemmatize(word))
    questions[index] = finalstr
    questions[index] = questions[index].strip()
#print("questions:", len(questions))

for index, question in enumerate(devQuestions):
    devQuestions[index] = question.lower() #lowercase
    devQuestions[index] = re.sub(r'\d+', '', devQuestions[index])  #remove numbers
    devQuestions[index] = devQuestions[index].translate(translator)
    finalstr = ""
    splitted = devQuestions[index].split(' ')
    for word in splitted:
        finalstr = finalstr + " " +porter.stem(lemmatizer.lemmatize(word))
    devQuestions[index] = finalstr
    devQuestions[index] = devQuestions[index].strip()
#print("devQuestions:",len(devQuestions))


#print(questions)
#print(devQuestions)

#tdidf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.4)


X_train = tfidfconverter.fit_transform(questions).toarray()
X_test = tfidfconverter.transform(devQuestions).toarray()

y_test = devLabels
y_train = labels


#classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=800, random_state=0)
classifier.fit(X_train, y_train) 

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
#from sklearn import svm
#classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
#classifier.fit(X_train,y_train)

#testing
y_pred = classifier.predict(X_test)
for elem in y_pred:
    print(elem)
#print("y_train: ",y_train)

#evaluating
from sklearn.metrics import accuracy_score
#print("y_test: ",y_test)
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
#print(accuracy_score(y_test, y_pred))