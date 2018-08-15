# Arbitrary runs:
# The purspose of this file is to test arbitrary python scripts/packages/and
# functions


import numpy as np
import chaospy as cp

def recursive_function(sigma, ref_collocation_pts, ref_collocation_w,colloc_xi_arr,
                        colloc_w_arr, actual_location, idx, ctr, index_list=[0,]):

    if idx == colloc_xi_arr.size-1:
        print "index_list = ", index_list
        sqrt2 = np.sqrt(2)
        for i in xrange(0, ref_collocation_pts.size):
            colloc_w_arr[idx] = ref_collocation_w[i] # Get the array of all the weights needed
            colloc_xi_arr[idx] = ref_collocation_pts[i] # Get the array of all the locations needed
            actual_location[ctr,:] = sqrt2*sigma*colloc_xi_arr
            print "ctr = ", ctr
            ctr += 1
        index_list.pop(-1)
        return idx-1, ctr
    else:
        print "index_list = ", index_list
        for i in xrange(0, ref_collocation_pts.size):
            colloc_xi_arr[idx] = ref_collocation_pts[i]
            colloc_w_arr[idx] = ref_collocation_w[i]
            index_list.append(idx+1)
            idx, ctr = recursive_function(sigma, ref_collocation_pts,
                                    ref_collocation_w,
                                    colloc_xi_arr, colloc_w_arr,
                                    actual_location,
                                    idx+1, ctr, index_list)
            # print "ctr = ", ctr
            # ctr+=1
        index_list.pop(-1)
        return idx-1, ctr


# Run the code
degree = 3
systemsize = 2
q,w = np.polynomial.hermite.hermgauss(degree)
colloc_xi_arr = np.zeros(systemsize)
colloc_w_arr = np.zeros(systemsize)
std_dev = np.ones(systemsize)
actual_location = np.zeros([degree**systemsize, systemsize])
ctr = 0
idx = 0

idx, ctr = recursive_function(std_dev, q, w,colloc_xi_arr, colloc_w_arr,
                              actual_location, idx, ctr)
