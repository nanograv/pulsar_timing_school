import mcmcmovie
import numpy as np

"""
Creating movies from an MCMC file is pretty simple. All that needs to be done is
to set the sample indices where we are going to make a movie frame.
"""

times = list(np.arange(0, 150, 1))
times += list(np.arange(150, 3000, 10))
times += list(np.arange(3000, 20000, 40))

mcmcmovie.make_movie('example-chain.txt', 'example.avi', inds=[0,1], burnin=0,
                     times=times, parlabels=['efac', 'Amp'], description='mcmc')
