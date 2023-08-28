'''The Below Program takes all the MFCC Files from the Train directory and merges them all together
for performing the K-means Algorithm on them to make the code book'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os



def main():


    dir =  '/home/tenet/Desktop/CS22Z121/Assignment_5/mfcc_train'
    digits = ['1','3','4','o','z']
    if not os.path.exists('kmeans_train'):
        os.makedirs('kmeans_train')
    kmeans_dir= 'kmeans_train'
    
    for digit in digits:
        output_dir = os.path.join(kmeans_dir,digit)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        data_dir = os.path.join(dir,digit)    
        data_merged = []
        for files in os.listdir(data_dir):
            if files.endswith('.mfcc'):
                data_merged.append(files)
        print(data_merged)
        

        with open(os.path.join(output_dir,'k_means.mfcc') , 'w') as output:
            for filename in data_merged:
                with open(os.path.join(data_dir,filename),'r') as input:
                    print(output.write(input.read()))      



if __name__ == '__main__' :
    main()