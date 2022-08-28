#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 10:59:58 2022

@author: Sandeep Kumar Suresh

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


reference_feature = pd.read_csv("/home/tenet/Documents/Speech Assignments/isolated_digits/2/1/jn_1.mfcc",skiprows=1, sep = ' ')

test_feature = pd.read_csv("/home/tenet/Documents/Speech Assignments/isolated_digits/2/4/ea_4.mfcc",skiprows=1, sep = ' ')


X = reference_feature.dropna(1).to_numpy()
Y = test_feature.dropna(1).to_numpy()

# print(X)

# print(Y)
def Distance_Matrix(A,B):
    m = A.shape[0]
    n = B.shape[0]
    #print(m,n)
    distance_matrix=np.zeros([m,n])
    
    for i in range(m):
        for j in range(n):
            #print(A[i],B[j])
            eu_distance = np.sqrt(np.sum(np.square(A[i]-B[j])))
           # print (np.sum(np.sqrt(abs(A[i]-B[j]))))
            distance_matrix[i,j] = eu_distance
            #distance_matrix[i,j] = np.sum(np.sqrt(A[i]-B[j]))
   # print(distance_matrix.shape)        
    return distance_matrix

def cost_matrix(distance_matrix):
    x,y = distance_matrix.shape
    c_matrix = np.zeros([x+1,y+1])
    print (c_matrix.shape)
    for i in range(1, x+1):
        c_matrix[i,0] = np.inf
    for j in range(1 , y+1):
        c_matrix[0,j] = np.inf
        
    
    for i in range(x):
        for j in range(y):
            
        
            dist = {
                    c_matrix[i,j],
                    c_matrix[i+1,j],
                    c_matrix[i,j+1]
                 }
            min_dist = min(dist)
            c_matrix[i+1,j+1] = distance_matrix[i,j] + min_dist    
        
    cost = c_matrix[x,y]/(x+y)  

    return cost      
            
    
        
        
x = Distance_Matrix(X, Y)

y=cost_matrix(x)        

print(y)
    


