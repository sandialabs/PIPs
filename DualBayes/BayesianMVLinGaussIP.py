import os, sys
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

class BayesianMVLinGaussIP: 
    def __init__(self,
                 prior_mean,
                 prior_cov,
                 dim,
                 noise_cov,
                 ):
        
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.dim = dim
        self.noise_cov = noise_cov
        # self.Gamma_post = self.posterior_calculations(self.A, self.y)

    # Function that computes the posterior mean and covariance
    def posterior_calculations(self, A=None, y=None):
        if A is None: 
            A = self.A
        Gamma_pri_inv = np.linalg.inv(self.prior_cov)
        Gamma_noise_inv = np.linalg.inv(self.noise_cov)

        Gamma_posterior = np.linalg.inv(A.T @ Gamma_noise_inv @ A + Gamma_pri_inv)

        mu_posterior = Gamma_posterior @ (A.T @ Gamma_noise_inv @ y + Gamma_pri_inv 
                                          @ self.prior_mean)
        
        #self.Gamma_posterior = Gamma_posterior
        return Gamma_posterior, mu_posterior
     
    #Functions that gets solutions and writes them to the class. 
    def get_problem_solutions(self):
         self.Gamma_posterior, self.mu_posterior = self.posterior_calculations()

    
    #Contour plot, take in the mean and variance/covariance
    def plot_distribution_contour(self,mean,cov,param_names = None):
    
        rv = ss.multivariate_normal(mean=mean, cov=cov)
        
        std = np.sqrt(np.diag(cov))


        x = np.linspace(mean[0] - 3*std[0], mean[0]+3*std[0], 100)
        y = np.linspace(mean[1]- 3*std[1] ,mean[1] + 3*std[1],100)

        
        X,Y = np.meshgrid(x,y)
        pos = np.dstack((X,Y))
        
        post_pdf = np.exp(rv.logpdf(pos))
        post_pdf[post_pdf==0] = np.nan

        # distr = rv.pdf(pos)

        plt.figure(figsize=(6,4))
        plt.contour(X,Y,post_pdf,levels = 8, colors = 'blue')
        if param_names is None:
            param_names = ['Parameter 1', 'Parameter 2']

        plt.xlabel(param_names[0])
        plt.ylabel(param_names[1])

        plt.title('Probability density contours')

        plt.tight_layout()
        plt.show()
        plt.close()
