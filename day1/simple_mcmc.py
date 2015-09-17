from __future__ import division
import numpy as np
import sys

class SimpleMCMC(object):
    
    def __init__(self, lnlikefn, lnpriorfn, sigmas):
        """
        Simple MCMC Sampler class that will perform
        sampling using gaussian jump proposals
        
        :param lnlikefun: ln-likelihood function.
        :param lnpriorfun: ln-prior function.
        :param sigmas: jump size (either single value or vector size of parameter array)
        
        """
        
        # collect input values
        self.lnlikefun = lnlikefn
        self.lnpriorfn = lnpriorfn
        self.sigmas = np.atleast_1d(sigmas)
        
    
    def sample(self, p0, N):
        """
        Sample for N steps.
        
        :param p0: Initial parameter array
        :param N: Number of iterations
        
        """
        
        ndim = len(p0)
        naccepted = 0
        
        # initialize arrays
        self.chain = np.zeros((N, ndim))
        self.lnlike, self.lnprob = np.zeros(N), np.zeros(N)
        
        # get likelihood and prior for initial point
        lnlike0, lnprior0 = self.lnlikefun(p0), self.lnpriorfn(p0)
        
        it = 0
        # start iterations
        for ii in range(N):
            
            # jump
            p1, qxy = self.jump(p0)
            
            # compute likelihoods and priors for new point
            lnprior1 = self.lnpriorfn(p1)
            if lnprior1 == -np.inf:
                lnlike1 = -np.inf
            else:
                lnlike1 = self.lnlikefun(p1)
            
            # hastings step
            diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0) + qxy
            if diff > np.log(np.random.rand()):

                # accept jump
                p0, lnlike0, lnprior0 = p1, lnlike1, lnprior1
                naccepted += 1
                
            # update arrays
            self.chain[ii,:] = p0
            self.lnlike[ii], self.lnprob[ii] = lnlike0, (lnlike0 + lnprior0)

            
            if it % 100 == 0 and it > 0:
                sys.stdout.write('\r')
                sys.stdout.write('Finished %g percent. Acceptance Rate: %g'%(it / N * 100, naccepted / it * 100))
                sys.stdout.flush()
            
            it += 1
            
    
    def jump(self, p0):
        
        """
        Standard gaussian jump
        
        :param p0: Current parameter vector
        
        :returns: new parameter vector, transition probability
        """
        
        qxy = 0
        q = p0.copy()

        # step sizes
        probs = [0.05, 0.7, 0.25]
        sizes = [0.1, 1.0, 10.0]
        scale = np.random.choice(sizes, p=probs)
        
        q += np.random.randn(len(q)) * self.sigmas * scale
        
        return q, qxy
