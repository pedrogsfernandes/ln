# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
    
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#open files
fTrain=open("training/TRAIN.txt", "r")
fQuestions=open("training/DEV-questions.txt", "r") #test file

#separating questions in test set
testLines = fQuestions.readlines()
devQuestions = []

for l in testLines:
    devQuestions.append(l)

#separating questions from targets

lines = fTrain.readlines()
labels = []
questions = []

for l in lines:
    lSplit = l.split(' ',1)
    labels.append(lSplit[0])
    questions.append(lSplit[1])

fTrain.close()
print(labels)
print(questions)

#preprocess text
import re
import string
translator = str.maketrans('', '', string.punctuation)

for index, question in enumerate(questions):
    questions[index] = question.lower() #lowercase
    questions[index] = re.sub(r'\d+', '', questions[index])  #remove numbers
    questions[index] = questions[index].translate(translator)
    questions[index] = questions[index].strip()
print(questions)


#stemming TODO

#lemmatization TODO

#countVector
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(questions)
count_vector=cv.fit_transform(questions)
#print(cv.vocabulary_)
#print(count_vector.shape)
#print(count_vector)

#tfidf
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
train = tfidfconverter.fit_transform(count_vector).toarray()
#print(train)

#training
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(train, labels) 

#testing


y_pred = classifier.predict(devQuestions)

#evaluating
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))