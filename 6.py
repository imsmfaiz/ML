import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metric

msg = pd.read_csv('naivetrext1.csv', names=['message', 'label'])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

cv = CountVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)
xtest_dtm = cv.transform(xtest)

print(cv.get_feature_names())
predicted = MultinomialNB().fit(xtrain_dtm, ytrain).predict(xtest_dtm)

print('Accuracy of the classfier is',metrics.accuracy_score(ytest,predicted))
print('Confusion matrix',metrics.confusion_matrix(ytest,predicted))
print('Recall & precision',metrics.precision_score(ytest,predicted))

