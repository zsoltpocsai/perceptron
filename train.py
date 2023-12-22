import numpy as np

def perceptron(lbc, data):
    w, b = lbc.get_params()
    X = data['x']
    Y = data['y']
    for i in range(len(X)):
        error = Y[i] - lbc.classify(X[i])
        w = np.add(w, np.multiply(X[i], error))
        b = b + error
        lbc.set_params(w, b)
