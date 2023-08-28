import os
from sklearn.metrics import accuracy_score
import numpy as np

test_seq_path = '/home/tenet/Desktop/CS22Z121/Assignment_5/K=100/obs_seq_test'
train_seq_path = '/home/tenet/Desktop/CS22Z121/Assignment_5/K=100/train_seq_hmm'

# train_hmm = ['1.seq.hmm','3.seq.hmm','4.seq.hmm','z.seq.hmm','o.seq.hmm']
digits = ['1','3','4','o','z']
tests,tests_label,train_seq = [],[],[]
for digit in digits:
    # print(digit)
    test_data_path = os.path.join(test_seq_path,digit)
    for files in os.listdir(test_data_path):
        # print(files)
        tests.append(files)
        tests_label.append(digit)
        # print(tests)
        # print(tests_label)
for i in os.listdir(train_seq_path):
# print(digit)
# print(files)
# print(i)

    train_seq.append(i)


pred_label = []

for test_file in tests:
    alpha_value = []
    train_labels = []

    for train_file in train_seq:
        shell_command ='./test_hmm ' + './K=100/test_seq/' + test_file + ' ./K=100/train_seq_hmm/' + train_file
        print(shell_command)
        os.system(shell_command)
        
        alpha_file = open('./alphaout', 'r')
        alpha_value.append(float(alpha_file.read().strip()))
        alpha_file.close()

        # print(train_file.split('/')[-1].split('.')[0])
        train_labels.append(train_file.split('/')[-1].split('.')[0])

    max_alpha_index = alpha_value.index(max(alpha_value))
    pred_label.append(train_labels[max_alpha_index])
    
accuracy = accuracy_score(tests_label, pred_label)
print(accuracy)
print("act:",tests_label)
print("pre:",pred_label)



















