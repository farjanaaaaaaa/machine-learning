
import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel


# # train scikit learn model
# clf = LogisticRegression()
# clf.fit(X_train,Y_train)
# print ('score Scikit learn: ', clf.score(X_test,Y_test))
#



##The sigmoid function adjusts the cost function hypotheses to adjust the algorithm proportionally for worse estimations
def logistic_func(z):
	pr_y = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return pr_y

##The hypothesis is the linear combination of all the known factors x[i] and their current estimated coefficients theta[i]
##This hypothesis will be used to calculate each instance of the Cost Function
def calc_hypothesis(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return logistic_func(z)

##For each member of the dataset, the result (Y) determines which variation of the cost function is used
##The Y = 0 cost function punishes high probability estimations, and the Y = 1 it punishes low scores
##The "punishment" makes the change in the gradient of ThetaCurrent - Average(CostFunction(Dataset)) greater
def calc_cost(X,Y,theta,m):
	sumOfErrors = 0
	for i in range(m):
		xi = X[i]
		hi = calc_hypothesis(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-hi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	# print ('cost is ', J )
	return J

##This function creates the gradient component for each Theta value
##The gradient is the partial derivative by Theta of the current value of theta minus
##a "learning speed factor aplha" times the average of all the cost functions for that theta
##For each Theta there is a cost function calculated for each member of the dataset
def partial_derivate_cost(X,Y,theta,j,m,alpha):
	errors_sum = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = calc_hypothesis(theta,X[i])
		error = (hi - Y[i])*xij # partial derivative w.r.t xij
		errors_sum += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * errors_sum
	return J

##For each theta, the partial differential
##The gradient, or vector from the current point in Theta-space (each theta value is its own dimension) to the more accurate point,
##is the vector with each dimensional component being the partial differential for each theta value
def gradient_descent(X,Y,theta,m,alpha):
	theta_new = []
	for pos_i in range(len(theta)):
		CFDerivative = partial_derivate_cost(X,Y,theta,pos_i,m,alpha)
		updated_theta = theta[pos_i] - CFDerivative
		theta_new.append(updated_theta)
	return theta_new

##The high level function for the LR algorithm which, for a number of steps (num_iters) finds gradients which take
##the Theta values (coefficients of known factors) from an estimation closer (new_theta) to their "optimum estimation" which is the
##set of values best representing the system in a linear combination model
def logistic_regression(X,Y,alpha,theta,num_iters):
	m = len(Y)
	for x in range(num_iters):
		new_theta = gradient_descent(X,Y,theta,m,alpha)
		theta = new_theta
		if x % 100 == 0:
			#here the cost function is used to present the final hypothesis of the model in the same form for each gradient-step iteration
			calc_cost(X,Y,theta,m)
	return theta


# These are the initial guesses for theta as well as the learning rate of the algorithm
# A learning rate too low will not close in on the most accurate values within a reasonable number of iterations
# An alpha too high might overshoot the accurate values or cause irratic guesses
# Each iteration increases model accuracy but with diminishing returns,
# and takes a signficicant coefficient times O(n)*|Theta|, n = dataset length

if __name__ == '__main__':
	df = pd.read_csv("data.csv")

	# clean up data
	x = df["label"].map(lambda x: float(str(x).rstrip(';')))
	X = df[["grade1", "grade2"]]
	X = np.array(X)
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
	X = min_max_scaler.fit_transform(X)
	Y = df["label"].map(lambda x: float(str(x).rstrip(';')))
	Y = np.array(Y)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

	initial_theta = [0,0] # Initial guess
	alpha = 0.1 # learning rate
	iterations = 1000 # Number of iterations
	optimal_theta = logistic_regression(X_train,Y_train,alpha,initial_theta,iterations)
	# Testing the model for accuracy
	score = 0
	length_test = len(X_test)
	for i in range(length_test):
		prediction = round(calc_hypothesis(X_test[i],optimal_theta))
		answer = Y_test[i]
		if prediction == answer:
			score += 1
	#the same process is repeated for the implementation from this module and the scores compared to find the higher match-rate
	my_score = float(score) / float(length_test)
	print('Accuracy: {0}%'.format(round(my_score*100, 2)))


	# visualize the test data with respective classes
	pos = where(Y_test == 1)
	neg = where(Y_test == 0)
	scatter(X_test[pos, 0], X_test[pos, 1], marker='o', c='b')
	scatter(X_test[neg, 0], X_test[neg, 1], marker='x', c='r')
	xlabel('Exam 1 score')
	ylabel('Exam 2 score')
	legend(['Not Admitted', 'Admitted'])
	show()