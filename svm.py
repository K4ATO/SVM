import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csvFile = "/home/kato/fcai/ML/Assignment2/part2/heart.csv"
data = pd.read_csv(csvFile)
data = data.sample(frac=1).reset_index(drop=True)


# a) preparing data and splitting
def prepdata(data, variable1, variable2):
    # separate X (training data) from y (target variable)
    Xtrain = data.loc[:(len(data) * 80 / 100), [variable1, variable2]]
    Ytrain = data.loc[:(len(data) * 80 / 100), ['target']]

    # normalizing the data
    Xtrain = (Xtrain - Xtrain.mean()) / Xtrain.std()
    # convert to matrices
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    Xtest = data.loc[(len(data) * 80 / 100):, [variable1, variable2]]
    Ytest = data.loc[(len(data) * 80 / 100):, ['target']]
    # normalizing the data
    Xtest = (Xtest - Xtest.mean()) / Xtest.std()
    # convert to matrices
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)

    f1_train = Xtrain[:, 0]
    f1_train = f1_train.reshape(Xtrain.shape[0], 1)

    f2_train = Xtrain[:, 1]
    f2_train = f2_train.reshape(Xtrain.shape[0], 1)

    f1_test = Xtest[:, 0]
    f1_test = f1_test.reshape(Xtest.shape[0], 1)
    f2_test = Xtest[:, 1]
    f2_test = f2_test.reshape(Xtest.shape[0], 1)

    return Xtrain, Ytrain, f1_train, f2_train, f1_test, f2_test, Ytest, Xtest


def train(X, Y, alpha, iterations):
    # initialize w1 and w2
    w1 = np.zeros((X.shape[0], 1))
    w2 = np.zeros((X.shape[0], 1))

    global cost
    for i in range(1, iterations):
        y = w1 * feature1 + w2 * feature2
        multp = y * Y
        count = 0
        for v in multp:
            if v >= 1:
                cost = 0
                w1 = w1 - alpha * (2 * 1 / i * w1)
                w2 = w2 - alpha * (2 * 1 / i * w2)

            else:
                cost = 1 - v
                w1 = w1 + alpha * (feature1[count] * Y_train[count] - 2 * 1 / i * w1)
                w2 = w2 + alpha * (feature2[count] * Y_train[count] - 2 * 2 / i * w2)

            count += 1
    return w1, w2, cost


def predict(w1, w2, f1, f2):
    index = list(range(60, 243))
    W1 = np.delete(w1, index)
    W1 = W1.reshape(60, 1)
    W2 = np.delete(w2, index)
    W2 = W2.reshape(60, 1)
    y_pred = W1 * f1 + W2 * f2
    return y_pred


# d)
def accuracy(y_pred, y_test):
    preds = []
    for pred in y_pred:
        if pred >= 0.5:
            preds.append(1)
        else:
            preds.append(0)
    count = 0
    percentage = 0
    for value in preds:
        if value == y_test[count]:
            percentage += 1
        count += 1
    accuracy_per = percentage / 60
    return accuracy_per


iterations = 1000
alpha = 0.0001

# b) different features
X_train, Y_train, feature1, feature2, f1_test, f2_test, Y_test, X_test = prepdata(data, 'trestbps', 'ca')

W1, W2, cost = train(X_train, Y_train, alpha, iterations)

Y_pred = predict(W1, W2, f1_test, f2_test)

accur = accuracy(Y_pred, Y_test)
print('svm Accuracy: ',accur)


# C) different learning rates
# alpha = 0.003
# W1_1, W2_1, cost1 = train(X_train, Y_train, alpha, iterations)
#
# alpha = 0.1
# W1_2, W2_2, cost2 = train(X_train, Y_train, alpha, iterations)
#
# alpha = 0.03
# W1_3, W2_3, cost3 = train(X_train, Y_train, alpha, iterations)
#
# alpha = 0.001
# W1_4, W2_4, cost4 = train(X_train, Y_train, alpha, iterations)
