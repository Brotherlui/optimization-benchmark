"""
This is a module containing a test of the gradient method.
"""


import numpy as np
from optimizers import StupidGradientDescent
from functions import Rosenbrock

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use("seaborn")


BUDGET = 1_000


def main():
    # initialize the benchmark function object
    fr = Rosenbrock()

    # this is making sure it is deterministic randomness
    np.random.seed(seed=42)

    # initialize optimization algorithm object
    initstate = 10 * (np.random.rand(2, 1) - 1)[:, 0]
    gd = StupidGradientDescent(0.0001, initstate, fr.grad)

    # numpy array for logging
    performance = np.zeros((BUDGET,))

    for i in range(BUDGET):
        # take a radient step with constant alpha
        gd.step()

        # evaluate the function at the new point in parameter space
        # print(f"\r{fr.eval(gd.state[0], gd.state[1])}  {gd.state}", end="")
        performance[i] = fr.eval(gd.state[0], gd.state[1])

    # print()

    plt.figure(figsize=(10, 10))
    plt.plot(np.log10(performance))
    # plt.plot(performance)
    plt.show()


if __name__ == "__main__":
    main()
