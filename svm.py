import random, csv
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

# Considering all elements in the data 
# are floats otherwise through error
def loadData(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

# This methods sets the data for the model 
# So it will be changes according to the data structure and types.
def train_test_data(dataset, splitRatio):
    trainData = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainData:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    X_train = [trainSet[i][:-1] for i in range(len(trainSet)) ]
    y_train = [trainSet[i][-1] for i in range(len(trainSet)) ]
    X_test = [copy[i][:-1] for i in range(len(copy)) ]
    y_test = [copy[i][-1] for i in range(len(copy)) ]
    return X_train, y_train, X_test, y_test


def main():
	filename = 'dataset.csv'
	splitRatio = 0.8 # 80% training and 20% testing
	dataset = loadData(filename)
	X_train, y_train, X_test, y_test = train_test_data(dataset, splitRatio)
	model = svm.SVC(kernel='linear' , C=1, gamma=1)
	model.fit(X_train, y_train)
	predicted = model.predict(X_test)
	accuracy = accuracy_score(y_test, predicted)
	confusion = confusion_matrix(y_test, predicted)
	print(accuracy)
	print(confusion)


if __name__ == '__main__':
    main()


