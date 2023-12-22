import numpy as np

class LinearBinaryClassifier:
    def __init__(self, size, shape=None):
        self.w = np.zeros(size)
        self.b = 0.0
        if shape is not None:
            self.w = np.reshape(self.w, shape)

    def get_params(self):
        return (self.w, self.b)

    def set_params(self, w, b):
        self.w = w
        self.b = b

    def classify(self, x):
        if (np.dot(self.w, x).sum() + self.b) > 0:
            return 1
        else:
            return 0

    def eval(self, data):
        num_of_errors = 0
        X = data['x']
        Y = data['y']
        for i in range(len(X)):
            if self.classify(X[i]) != Y[i]:
                num_of_errors += 1
        return num_of_errors / len(X)

    # only for size = 2
    def line(self, X):
        w = np.add(self.w, 0.001) # in case some w is 0
        Y = [ (x * w[0] + self.b) / -w[1] for x in X ]
        return X, Y
