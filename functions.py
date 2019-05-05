"""
This is a module containing functions and their gradients
"""

import numpy as np


class Rosenbrock:

    def __init__(self, a=1., b=100.):
        self.a = a
        self.b = b

    def eval(self, x, y):
        return (self.a - x)**2 + self.b * (y - x**2)**2

    def grad(self, x, y):
        return np.array([-2. * (self.a - x), 0.]) + 2. * self.b * (y - x**2.) * np.array([-2. * x, 1.])

    def evaluate(self, params):
        return (self.a - params[0])**2 + self.b * (params[1] - params[0]**2)**2

    def gradient(self, params):
        return np.array([-2. * (self.a - params[0]), 0.]) + 2. * self.b * (params[1] - params[0]**2.) * np.array([-2. * params[0], 1.])
