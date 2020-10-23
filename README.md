# Natural Language

Run the coarse model:
```
 python qc.py -coarse training/TRAIN.txt training/DEV-questions.txt > predicted-labels.txt

```

Run the fine model:
```
 python qc.py -fine training/TRAIN.txt training/DEV-questions.txt > predicted-labels.txt

```
Evaluate your results: (you can also uncomment the last line in qc.py)
```
python ./evaluate.py training/DEV-labels.txt predicted-labels.txt
```


a) Predict the labels for the data.
python qc.py -coarse TRAIN.txt DEV-questions.txt > predicted-labels.txt # runs the coarse model
python qc.py -fine TRAIN.txt DEV-questions.txt > predicted-labels.txt # runs the fine model
b) Evaluate the performance of your model using the true labels:
python ./evaluate.py DEV-labels.txt predicted-labels.txt