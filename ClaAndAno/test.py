
from sklearn.svm import OneClassSVM
X = [[0],[0.43],[0.44], [0.45], [0.46]]
clf = OneClassSVM(nu=0.01,gamma='auto').fit(X)
print(clf.predict(X))
print(clf.score_samples(X))
print(clf.decision_function(X))
