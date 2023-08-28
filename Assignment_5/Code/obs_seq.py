import numpy as np
import pandas as pd
import os

def observation_sequence_generator(data_mfcc , codebook_data):
    observation_seq = []
    for i in range(data_mfcc.shape[0]):
        distance  = []
        for j in range(codebook_data.shape[1]):
            eu_distance = np.sqrt(np.sum(np.square(data_mfcc.iloc[i] - codebook_data.iloc[:,j])))
            distance.append(eu_distance)
        min_dist = min(distance)
        index = distance.index(min_dist)
        observation_seq.append(index)
    return observation_seq

def main():
    codebook_path = '/home/tenet/Desktop/CS22Z121/Assignment_5/codebook100.csv'
    codebook_data = pd.read_csv(codebook_path)

    mfcc_train_path = '/home/tenet/Desktop/CS22Z121/Assignment_5/mfcc_train'
    mfcc_test_path = '/home/tenet/Desktop/CS22Z121/Assignment_5/mfcc_test'
    # data_mfcc_test = pd.read_csv('/home/tenet/Desktop/CS22Z121/Assignment_5/mfcc_train/1/ac_1.mfcc' , sep=' ' , header=None)
    # digits = ['1','3','4','o','z']
    digits = ['1']

    if not os.path.exists('observation_sequence_train'):
            os.makedirs('observation_sequence_train')
    output_dir =  'observation_sequence_train'
    
    # a = observation_sequence_generator(data_mfcc_test,codebook_data)
    # print(a)
    
    for digit in digits:
        if not os.path.exists('observation_sequence_train'):
            os.makedirs('observation_sequence_train')
        output_dir =  'observation_sequence_train'
        
        filepath = os.path.join(mfcc_train_path,digit)
        obs = []
        for files in os.listdir(filepath):
            print(files)
          
            data_mfcc = pd.read_csv(os.path.join(filepath,files) , sep =' ' , header=None)
            seq = observation_sequence_generator(data_mfcc,codebook_data)
            obs.append(seq)
        #print(obs)
        
        with open('1.seq', 'w') as input:
            for listitem in obs:
                input.write('%s\n' % listitem)
   
if __name__ == '__main__':
    main()            

