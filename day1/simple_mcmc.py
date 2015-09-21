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
        
        self.ndim = len(p0)
        naccepted = 0
        
        # initialize arrays
        self.chain = np.zeros((N, self.ndim))
        self.lnlike, self.lnprob = np.zeros(N), np.zeros(N)
        
        # get likelihood and prior for initial point
        lnlike0, lnprior0 = self.lnlikefun(p0), self.lnpriorfn(p0)
        
        self.iter = 0
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

            
            if self.iter % 100 == 0 and self.iter > 0:
                sys.stdout.write('\r')
                sys.stdout.write('Finished %g percent. Acceptance Rate: %g'%(self.iter / N * 100, naccepted / self.iter * 100))
                sys.stdout.flush()
            
            self.iter += 1
            
    
    def jump(self, p0):
        
        """
        Standard gaussian jump
        
        :param p0: Current parameter vector
        
        :returns: new parameter vector, transition probability
        """
        
        qxy = 0
        q = p0.copy()
        
        # 50/50 jup probability
        jumps = ['gaussian', 'de']
        jprobs = [0.5, 0.5]

        # draw jump type
        jname = np.random.choice(jumps, p=jprobs)
        
        # differential evolution
        if jname == 'de' and self.iter > 1000:

            # draw a random integer from 0 - iter
            mm = np.random.randint(0, self.iter)
            nn = np.random.randint(0, self.iter)

            # make sure mm and nn are not the same iteration
            while mm == nn:
                nn = np.random.randint(0, self.iter)

            # mode jump
            if np.random.rand() > 0.5:
                scale = 1.0

            else:
                scale = 2.4 / np.sqrt(2 * self.ndim)

            q += scale * (self.chain[mm,:] - self.chain[nn,:])

        # standard gaussian default
        else:
            # step sizes
            probs = [0.05, 0.7, 0.25]
            sizes = [0.1, 1.0, 10.0]
            scale = np.random.choice(sizes, p=probs)
            q += np.random.randn(len(q)) * self.sigmas * scale


        
        return q, qxy
