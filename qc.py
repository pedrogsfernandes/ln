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

#open files
fTrain=open("training/TRAIN.txt", "r")
fQuestions=open("training/DEV-questions.txt", "r") #test file
fLabels=open("training/DEV-labels.txt", "r") #test file

#separating questions in test set
testLines = fQuestions.readlines()
devQuestions = []
for l in testLines:
    devQuestions.append(l)

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
    questions.append(lSplit[1])

fTrain.close()
fQuestions.close()
fLabels.close()

import numpy 
questions = numpy.array(questions)
devQuestions = numpy.array(devQuestions)
print("labels:", len(labels))
print("questions:", len(questions))

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
print("questions:", len(questions))

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
print("devQuestions:",len(devQuestions))


#stemming TODO

#lemmatization TODO


#tdidf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.5)


X_train = tfidfconverter.fit_transform(questions).toarray()
X_test = tfidfconverter.transform(devQuestions).toarray()

y_test = devLabels
y_train = labels


#training
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 



#testing
y_pred = classifier.predict(X_test)
print("y_pred: ",y_pred)
print("y_train: ",y_train)

#evaluating
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("y_test: ",y_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))