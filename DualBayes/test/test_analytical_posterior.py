import pytest
import numpy as np
import sys
sys.path.append('../')
from BayesianMVLinGaussIP import *

def test_posterior():
    prior_mean = np.ones(3)
    prior_cov = np.eye(3)
    noise_cov = np.eye(3)
    A = np.eye(3)
    y = np.zeros(3)

    posterior_cov = 1/2 * np.eye(3)
    posterior_mean = posterior_cov @ prior_mean

    mypost = BayesianMVLinGaussIP( prior_mean=prior_mean,
                                  prior_cov=prior_cov, 
                                   dim=3, 
                                    noise_cov=noise_cov )
    computed_cov, computed_mean = mypost.posterior_calculations(A,y)

    assert (np.linalg.norm(posterior_mean - computed_mean) < 1e-15) and \
           ( np.linalg.norm(posterior_cov - computed_cov) < 1e-15 )