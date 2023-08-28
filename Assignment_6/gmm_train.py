import numpy as np
import random
import pandas as pd
import pickle
from scipy.stats import multivariate_normal as mp
import numpy as np
import pickle
from tqdm.auto import tqdm
import argparse
np.random.seed(2027)

class KMeans():
    def __init__(self,n_clusters,n_iterations):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.centroid = None

    def predict(self,X):
        random_index = random.sample(range(0,X.shape[0]),self.n_clusters)
        self.centroid = X[random_index]

        
        for i in tqdm(range(self.n_iterations)):
            cluster_group = self.assign_clusters(X)
            old_centroid = self.centroid
            self.centroid = self.move_centroids(X,cluster_group)
        
            # if (old_centroid == self.centroid).all():
            #     break
            #print('i b l',i)
            if i == self.n_iterations:
                print('i a l',i)
                break
        return cluster_group    


    def assign_clusters(self,X):

        cluster_group = []
        eu_distances = []
        
        for row in X:
            for centroids in self.centroid:
                eu_distances.append(np.sqrt(np.dot(row-centroids,row-centroids)))
            min_distance = min(eu_distances)
            index_pos = eu_distances.index(min_distance)
            cluster_group.append(index_pos)
            eu_distances.clear()
        return np.array(cluster_group)


    def move_centroids(self,X,cluster_group):
        
        new_centroids = []
        
        cluster_type = np.unique(cluster_group)
        
        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis = 0))
            
        return np.array(new_centroids)   
    
    def datapoint_array(self,X,cluster_group):
    
        index = np.unique(cluster_group)
        cluster_datapoints = []
        # sum = 0
        for idx in index:            
            datapoints = X[cluster_group == idx]
            # sum+=datapoints.shape[0]
            cluster_datapoints.append(datapoints)
        # print(sum)    
        return np.array(cluster_datapoints)




class EM():


    def __init__(self,clusters,data,mu,pi,sigma):
        self.clusters = clusters
        self.data =  data
        self.mu = mu
        self.pi = pi
        self.sigma = sigma
    
    def display(self):
        print('clusters',self.clusters)
        print('data',self.data.shape)
        print('mean',self.mu.shape)
        print('weights',self.pi.shape)
        print('covariance',self.sigma.shape)     

    
    # def covariance(self):
    #     n = data1.shape[0]
    #     # print(data1.shape)
    #     mu = np.mean(data1,axis=0)
    #     # print('mu',mu.shape)
    #     data1 = data1 - mu
    #     # print(data1)
    #     cov = np.dot(data1.T, data1) /(n-1)
    #     return cov
    
    # def diag(self):
    #     cov_matrix = np.zeros((cluster_Size,dim,dim))
    #     # print(w[0].shape)
    #     for it,i in enumerate(w):
    #         # print(i.shape)
    #         data2 = np.zeros((i.shape))
    #         for j,k in enumerate(i):
    #             # print(j.shape)
    #             data2[j]=k[np.newaxis,:]
    #         # print(data2.shape)
    #             # cov = np.vstack(j[np.newaxis,:])
    #         d = covariance(data2)
    #         # print(d.shape)   
    #         diag = np.zeros_like(d)
    #         # print(diag.shape)
    #         np.fill_diagonal(diag, np.diag(d))
    
   

    def E_Step(self):

        self.gamma = np.zeros((self.data.shape[0],self.clusters))
        for i in range(self.clusters):
            
            self.gamma[:,i] = self.pi[i] * mp.pdf(self.data ,self.mu[i],self.sigma[i,:,:]).ravel()


            # print(self.gamma[:,i])
        gamma_sum = np.sum(self.gamma, axis=1)[:,np.newaxis]
        self.gamma /= gamma_sum

        return self.gamma


    def M_Step(self):

        # print('gamma', self.gamma.shape)
        N = self.data.shape[0] # number of objects
        C = self.gamma.shape[1] # number of clusters
        d = self.data.shape[1] # dimension of each object

        N_k = np.sum(self.gamma, axis = 0)[:,np.newaxis]

        # responsibilities for each gaussian
        self.pi = np.mean(self.gamma, axis = 0)

        self.mu = np.dot(self.gamma.T, data) / N_k

        
        for i in range(C):
            gamma_outer_sum = 0
            for j in range(N):
                x_minus_mu = data[j,:] - self.mu[i]
                gamma_outer_sum += self.gamma[j,i]*np.outer(x_minus_mu,x_minus_mu)
                sigma_ND = gamma_outer_sum /N_k[i]
        
            np.fill_diagonal(self.sigma[i,:,:],np.diag(sigma_ND))
 

        return self.pi, self.mu, self.sigma


    def train_gmm(self, n_epochs):
        clusters = self.clusters
        
        for i in tqdm(range(n_epochs)):
            #print(i)
        
            self.E_Step()
            self.M_Step()

        #print(self.gamma)
        return self.pi , self.mu ,self.sigma





    



if __name__ == '__main__'    :


    parser = argparse.ArgumentParser()
    parser.add_argument("K",help = 'Enter the Cluster Size')

    args = parser.parse_args()
    



    cluster_Size = int(args.K)
    x = KMeans(cluster_Size,10)
    data = pd.read_csv('mfcc_zeromean_unit_variance/gmm_with_zmuv.mfcc' , sep = ',' , header =None).values
    y = x.predict(data)
    mu = x.move_centroids(data,y)
    w = x.datapoint_array(data,y)


   

    def pi_calculation( cluster_datapoints):
        N = data.shape[0]
        init_pi =[]
        for i in cluster_datapoints:
            x = i.shape[0]/N
            init_pi.append(x)
        return np.array(init_pi)


    def covariance(data1):
        n = data1.shape[0]
        # print(data1.shape)
        mu = np.mean(data1,axis=0)
        # print('mu',mu.shape)
        data1 = data1 - mu
        # print(data1)
        cov = np.dot(data1.T, data1) /(n-1)
        return cov

    dim = data.shape[1]    
    cov_matrix = np.zeros((cluster_Size,dim,dim))
    # print(w[0].shape)
    for it,i in enumerate(w):
        # print(i.shape)
        data2 = np.zeros((i.shape))
        for j,k in enumerate(i):
            # print(j.shape)
            data2[j]=k[np.newaxis,:]
        # print(data2.shape)
            # cov = np.vstack(j[np.newaxis,:])
        d = covariance(data2)
        # print(d.shape)   
        diag = np.zeros_like(d)
        # print(diag.shape)
        np.fill_diagonal(diag, np.diag(d))
        
        
    
        cov_matrix[it,:,:] = diag 

    

    def MAP_Adaptation(speaker_data_path):
        """Doing one round of EM Algorithm for Adapting the Speaker Data """
        speaker_data = pd.read_csv(speaker_data_path,sep=' ',header=None)
        # speaker_data = speaker_data.drop(columns=0,axis=1)


        ubm_model = pickle.load(open(f'gmm_speaker_ubm_{cluster_Size}_zmuv.pkl' ,'rb'))
        N,D = speaker_data.shape
        # print('N',N , 'D',D)
        K = int(args.K)
        # print(K)
        w_ubm = ubm_model[0]
        mu_ubm = ubm_model[1]
        sigma_ubm = ubm_model[2]

        """For Now Writing the E_Step Algorithm Code here and then changing the Code Structure of K-Means"""
        gamma_nk = np.zeros((N,K))
        for i in range(0,K):
        
            # gamma_nk[:,i] = np.log(w_ubm[i]) * mp.logpdf(speaker_data ,mu_ubm[i],sigma_ubm[i,:,:]).ravel()
            gamma_nk[:,i] = (w_ubm[i]) * mp.pdf(speaker_data ,mu_ubm[i],sigma_ubm[i,:,:]).ravel()


        N_k = np.sum(gamma_nk,axis=0)
        gamma_nk /= N_k




        mu_updated = []
        for i in range(K):
            mu_updated.append(np.dot(gamma_nk[:,i],speaker_data))   
        mu_upd = np.stack(mu_updated,axis=0)
        # print('mu_up',mu_upd.shape)
        
        r =16
        alpha_k = N_k/(N_k + r)
        # print('alpha_k_1',alpha_k.shape)

        alpha_k = alpha_k.reshape(-1, 1)

        mu_new = alpha_k*mu_upd+(1-alpha_k)*mu_ubm
 
        """It came to my notice that the mu values are not adapting
        Possibilities are:
                        1.alpha value is small and (1- alpha)value is large so the mu_new = mu_ubm   --->took ZMUV
                        2.Why is the alpha value small-->N-K-->gamma-->ubm-->all program is wrong :)
                        3.Gamma value was low in E_Step and M_step , used that value to calculate parameters
                        
                Fix     Took Zero Mean Unit Variance to bring the MFCC Feature Vectors and fixed all the problems 
                            including the Accuracy   """
        

        

        return mu_new , sigma_ubm



    weights = pi_calculation(w)
    a = EM(cluster_Size,data,mu,weights,cov_matrix)
    train = a.train_gmm(3)
    pickle.dump(train,open(f'gmm_speaker_ubm_{cluster_Size}_zmuv.pkl' ,'wb'))


    # The Above Code does the creates and UBM Model and dumps the Model to the Required File Directory

    """The Next Step is to use the UBM Model to Adapt the Speaker and Compute the Accuracy
    
    I have done the rest of code in jupyter notebook i.e. after dumping the model , created 
    Adapted Train Speakers and Test Speakers and Computed the Accuracy
    
    
    """





