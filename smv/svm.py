import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

# Two types of cancer being:

classes = ['malignant' 'benign']

# c is soft margin
clf = svm.SVC(kernel="linear", C=1)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(y_pred)
