fLabels=open("DEV-labels.txt", "w+")
fQuestions=open("DEV-questions.txt", "w+")

fDev=open("DEV.txt", "r")
lines = fDev.readlines()

for l in lines:
	lSplit = l.split(' ',1)
	fLabels.write(lSplit[0]+'\n')
	fQuestions.write(lSplit[1])

fDev.close()
fLabels.close()
fQuestions.close()