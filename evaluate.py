import sys

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

#open files
fAnswer=open(sys.argv[1], "r")
fPredict=open(sys.argv[2], "r") #test file


answerLabels = []
for l in fAnswer.readlines():
    answerLabels.append(l.replace('\n',''))

predictionLabels = []
for l in fPredict.readlines():
    predictionLabels.append(l.replace('\n',''))


from sklearn.metrics import accuracy_score
print(accuracy_score(answerLabels, predictionLabels))

fAnswer.close()
fPredict.close()