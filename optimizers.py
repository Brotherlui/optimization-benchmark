"""
This is a module containing optimizers
"""


import numpy as np


class StupidGradientDescent:

    def __init__(self, alpha, initstate, fgrad):
        self.state = initstate
        self.alpha = alpha
        self.fgrad = fgrad

    def step(self):
        self.state -= self.alpha * self.fgrad(self.state[0], self.state[1])
