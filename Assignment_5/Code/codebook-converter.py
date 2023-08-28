
# importing libraries   
from sklearn.cluster import KMeans  
import numpy as nm
import numpy as np    
import matplotlib.pyplot as mtp    
import pandas as pd    
# Importing the dataset  
x = pd.read_csv('/home/tenet/Desktop/CS22Z121/Assignment_5/kmeans_train/k-means-train.mfcc',sep=' ',header=None)  


#training the K-means model on a dataset  
cluster = 5
kmeans = KMeans(n_clusters=cluster, init='k-means++', random_state= 50)
y_predict= kmeans.fit_predict(x)  
centres=kmeans.cluster_centers_
df1=pd.DataFrame(centres)
df2=pd.DataFrame(np.arange(cluster))
code_book=pd.concat([df1,df2],axis=1)
code_book.T.to_csv('codebook5.csv',index= False)

# print(code_book.shape)