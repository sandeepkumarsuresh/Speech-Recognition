import numpy as np
import pandas as pd
import os
import pathlib

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

    # mfcc_train_path = '/home/tenet/Desktop/CS22Z121/Assignment_5/mfcc_train'
    mfcc_test_path = '/home/tenet/Desktop/CS22Z121/Assignment_5/mfcc_test'
    # data_mfcc_test = pd.read_csv('/home/tenet/Desktop/CS22Z121/Assignment_5/mfcc_train/1/ac_1.mfcc' , sep=' ' , header=None)
    digits = ['1','3','4','o','z']
    # digits = ['z']

    # a = observation_sequence_generator(data_mfcc_test,codebook_data)
    # print(a)
    
    for digit in digits:
        if not os.path.exists('obs_seq_test'):
            os.makedirs('obs_seq_test')
        output_dir =  'obs_seq_test'
        
        output_path = os.path.join(output_dir,digit)
        filepath = os.path.join(mfcc_test_path,digit)
        
        for files in os.listdir(filepath):
            obs = []
            print(files)
          
            data_mfcc = pd.read_csv(os.path.join(filepath,files) , sep =' ' , header=None)
            file_prefix = pathlib.Path(files).stem
            seq = observation_sequence_generator(data_mfcc,codebook_data)
            obs.append(seq)
            o_file_name = "{}.seq".format(file_prefix)
            o_file_path = os.path.join(output_dir,o_file_name)
            o_file = open(o_file_path,'w')
            # o_file.write(str(mfcc_audio))
            o_file.write(str(obs))
            # o_file.write(str(np.ceil(np.savetxt(o_file,obs,delimiter=' '))))
            o_file.close()
            print(o_file_path)

            
       

if __name__ == '__main__':
    main()            







































 