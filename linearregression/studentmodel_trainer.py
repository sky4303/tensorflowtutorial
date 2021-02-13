import numpy
import pandas
import sklearn as skl
import sklearn.linear_model as linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pandas.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "absences", "failures", "studytime"]]

predict = "G3"
X = numpy.array(data.drop([predict], 1))  # everything but G3
Y = numpy.array(data[predict])  # only G3
x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(X, Y, test_size=0.1)

"""best = 0
for i in range(40):
    x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

accuracy = linear.score(x_train, y_train)
#print(accuracy)
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

style.use("ggplot")
p = "absences"
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade (G3)")
#pyplot.show()

