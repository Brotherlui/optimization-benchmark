"""
This is a module containing a test of the gradient method.
"""


import numpy as np
import optimizers
from functions import Rosenbrock

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use("seaborn")


BUDGET = 1000


def main():
    # initialize the benchmark function object
    fr = Rosenbrock()

    # this is making sure it is deterministic randomness
    np.random.seed(seed=42)

    # initialize optimization algorithm object
    initstate = 10 * (np.random.rand(2, 1) - 1)[:, 0]
    # gd = optimizers.StupidGradientDescent(0.0001, initstate, fr.grad)
    gd = optimizers.SmarterGradientDescent(0.0001, initstate, fr.gradient)
    mgd = optimizers.Momentum_GD(0.0001, 0.001, initstate, fr.gradient)

    # numpy array for logging
    performance_gd = np.zeros((BUDGET,))
    performance_mgd = np.zeros((BUDGET,))

    for i in range(BUDGET):
        # take a radient step with constant alpha
        gd.step()
        mgd.step()

        # evaluate the function at the new point in parameter space
        # performance[i] = fr.eval(gd.state[0], gd.state[1])
        performance_gd[i] = fr.evaluate(gd.state)
        performance_mgd[i] = fr.evaluate(mgd.state)

    # print()

    plt.figure(figsize=(10, 10))
    plt.plot(np.log10(performance_gd))
    plt.plot(np.log10(performance_mgd))
    plt.savefig("rosenbrock.png", dpi=200)


if __name__ == "__main__":
    main()
