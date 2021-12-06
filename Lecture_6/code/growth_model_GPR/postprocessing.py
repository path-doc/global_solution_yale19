#======================================================================
#
#     This module contains routines to postprocess the VFI 
#     solutions.
#
#     Simon Scheidegger, 01/19
#     Cameron Gordon, 11/21 - updates to Python3 (print statements + pickle)
#======================================================================

import numpy as np
from parameters import *
#import cPickle as pickle
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import nonlinear_solver_iterate as solver
import os
#======================================================================    
# Routine compute the errors
def ls_error(n_agents, t1, t2, num_points):
    file=open('errors.txt', 'w')
    
    np.random.seed(0)

    dim = n_agents
    Xtraining = np.random.uniform(k_bar, k_up, (No_samples, dim))
    to_print=np.empty((1,5))
        
    for i in range(t1, t2-1):
        sum_diffs=0
        diff = 0
      
        # Load the model from the previous iteration step
        restart_data = filename + str(i) + ".pcl"
        with open(restart_data, 'rb') as fd_old:
            gp_old = pickle.load(fd_old)
            print("data from iteration step ", i , "loaded from disk")
        fd_old.close()      

        # Load the model from the previous iteration step
        restart_data = filename + str(i+1) + ".pcl"
        with open(restart_data, 'rb') as fd:
            gp = pickle.load(fd)
            print("data from iteration step ", i+1 , "loaded from disk")
        fd.close()
      
        mean_old, sigma_old = gp_old.predict(k_sample, return_std=True)
        mean, sigma = gp.predict(k_sample, return_std=True)

        gp_old = gp
        targ_new = solver.iterate(k_sample, n_agents, gp_old)[0]
        # plot predictive mean and 95% quantiles
        #for j in range(num_points):
            #print k_sample[j], " ",y_pred_new[j], " ",y_pred_new[j] + 1.96*sigma_new[j]," ",y_pred_new[j] - 1.96*sigma_new[j]

        diff_mean = mean_old - mean
        max_diff_mean = np.amax(np.fabs(diff_mean))
        avg_diff_mean = np.average(np.fabs(diff_mean))

        diff_targ = mean - targ_new
        max_diff_targ = np.amax(np.fabs(diff_targ))
        avg_diff_targ = np.average(np.fabs(diff_targ))

        to_print[0,0]= i+1
        to_print[0,1]= max_diff_mean
        to_print[0,2]= avg_diff_mean
        to_print[0,3]= max_diff_targ
        to_print[0,4]= avg_diff_targ
        
        np.savetxt(file, to_print, fmt='%2.16f')
        msg = "Cauchy:" + str(diff_mean) + ", max = " + str(round(max_diff_mean,3))
        msg += os.linesep()
        msg += "Absolute:" + str(diff_targ) + ", max = " + str(round(max_diff_targ,3))
        print(msg)
        print("===================================")

        
    file.close()
    
    return 
        
#======================================================================
