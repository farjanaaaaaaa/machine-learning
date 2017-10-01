# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math, sys


def loadData(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def train_test_data(dataset, splitRatio):
    trainData = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainData:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def splitByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def statistics_continuous(numbers):
    mean = sum(numbers) / float(len(numbers))
    variance = sum([pow(x - mean, 2) for x in numbers]) / float(len(numbers) - 1)
    return [sum(numbers) / float(len(numbers)), math.sqrt(variance)]

def statistics_discrete(feature):
    n = len(feature)
    vals = list(set(feature))
    feature_stats = {}
    for val in vals:
        n_i = feature.count(val)
        p_i = n_i/n # probability for a specific object
        feature_stats[val] = (p_i)
    return feature_stats



def summarize(dataset, categoricalList=[]):
    unpack_data = [x for x in zip(*dataset)]
    summaries = []
    if categoricalList != []:
        x = 0
        for cat in categoricalList:
            summaries.insert(cat, statistics_discrete(unpack_data[cat]))
            # summaries.insert(cat, ())
    continuous_var = list(set([x for x in range(len(unpack_data))]) - set(categoricalList))
    continuous_data = [x for i, x in enumerate(unpack_data) if i in continuous_var]

    for i in range(len(continuous_data)):
        mean, stdev = statistics_continuous(continuous_data[i])
        summaries.insert(continuous_var[i], (mean, stdev))
    del summaries[-1] # Decision variable must be the last column of the dataset
    return summaries



def summarizeByClass(dataset, categoricals):
    separated = splitByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances, categoricals) # put here [] is the index of categorical variable(s)
    return summaries


def estimateProbability_gaussian(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            try:
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= estimateProbability_gaussian(x, mean, stdev)
            except ValueError:
                category = inputVector[i]
                try:
                    probability_categorical = classSummaries[i][category]
                    probabilities[classValue] *= probability_categorical
                except KeyError:
                    probabilities[classValue] *= 0.0001
                    ''' 0.0001 is hard coded for simplicity but Python throws KeyError when it does not find any
                    Key so it means the category has not occured Yet. So in this case a reasonable estimate
                    would be to divide the new occurence with the training set size N because out N trials
                    we have not observed any of that occurence so it would the first one.'''
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    filename = 'pima-diabetes-dataset-with-categorical-var.csv'
    splitRatio = 0.7 # 70% training and 30% testing
    dataset = loadData(filename)
    trainingSet, testSet = train_test_data(dataset, splitRatio)
    categorical_variables = [0,8]
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet, categorical_variables)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))


main()