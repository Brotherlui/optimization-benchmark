"""
This is a module containing optimizers
"""


import numpy as np


class StupidGradientDescent:

    def __init__(self, alpha, state, fgrad):
        self.state = state
        self.alpha = alpha
        self.fgrad = fgrad

    def step(self):
        self.state -= self.alpha * self.fgrad(self.state[0], self.state[1])


class SmarterGradientDescent:

    def __init__(self, alpha, state, fgrad):
        self.state = state
        self.alpha = alpha
        self.fgrad = fgrad

    def step(self):
        self.state -= self.alpha * self.fgrad(self.state)


class Momentum_GD:

    def __init__(self, alpha, gamma, state, fgrad):
        self.state = state
        self.alpha = alpha
        self.fgrad = fgrad
        self.gamma = gamma
        self.past_step = 0

    def step(self):
        new_step = self.gamma * self.past_step + self.alpha * self.fgrad(self.state)
        self.state = self.state - new_step
