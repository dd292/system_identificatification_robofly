from Data_interpret.GroundTruth import ReadMat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy
from scipy import sparse

class GP:

    def __init__(self, delta_t= 1/100, traj_length=1, num_epoch= 7):
        self.DELTA_T =delta_t
        self.TRAJECTORY_LENGTH = traj_length#seconds
        self.NUM_TRAINING_EPOCHS= num_epoch
        self.kernel_length_scales = np.ones((12,9))*1.9
        self.kernel_scale_factors = np.ones((9,1))*0.01
        self.noise_sigmas = np.array([.05,.05,.05,.05,.05,.05,.05,.05,.05])*0.05
        self.inv_value =None
        self.K = None
    def make_training_data(self,epoch, state_traj, action_traj):
        start_point= int(epoch*self.TRAJECTORY_LENGTH/self.DELTA_T)
        end_point = int((epoch+1)*self.TRAJECTORY_LENGTH/self.DELTA_T)
        cur_state_traj= state_traj[:,start_point:end_point]
        cur_action_traj= action_traj[:,start_point:end_point]
        delta_state_traj= cur_state_traj[:,1:]- cur_state_traj[:,:-1]
        x= np.concatenate((cur_state_traj[:,:-1],cur_action_traj[:,:-1]),axis=0)
        return x,delta_state_traj,cur_action_traj
    def get_pce_kernel_fast(self,X, X_prime, Gp_number,partial= False):
        K= np.zeros((X.shape[0],X_prime.shape[0]))
        sigma_f= self.kernel_scale_factors[Gp_number]
        l= self.kernel_length_scales[:, Gp_number];
        for iter1,i in enumerate(X):
            if not partial:
                noise= sigma_f**2
            else:
                noise= 2*np.sqrt(sigma_f)
            K [iter1,:] = noise * np.exp(-np.sum((i-X_prime)**2/(2*l**2),axis=1))
        #print(K.shape)
    
        return K
    def get_inv_val(self,train_x,train_y,func):
        inv_value = np.zeros((train_y.shape[1],train_x.shape[0],train_x.shape[0]))
        K= np.zeros((train_x.shape[0],train_x.shape[0], train_y.shape[1]))
        for k in range(train_y.shape[1]):

            K[:,:, k] = func(train_x, train_x, k)
            inv_value[k,:] = np.linalg.inv(K[:,:,k]+ self.noise_sigmas[k]**2*np.eye(train_x.shape[0]))
        self.inv_value= inv_value
        self.K= K
        return inv_value
    def update_hyperparams(self,train_y):
        for i in range(3):
            l=1
            #dkdth[i]=
        for i in range(3): #because there are 3 parameters
            for i in range(train_y.shape[1]):
                log_liklihood= 0.5* np.matrix.trace(np.linalg.inv(self.K[:,:,i])@train_y[:,i]@ dkdth[i])
                print('log_liklihood',log_liklihood.shape)
                S= log_liklihood @ log_liklihood.T
                print('s',S.shape)
    def predict_gp(self,train_x,train_y,init_state, action_traj,inv_val):
    
        output_dimension= train_y.shape[1]
    
        number_of_training_points=int(self.TRAJECTORY_LENGTH/self.DELTA_T)
        pred_gp_mean = np.zeros((number_of_training_points, output_dimension))
        pred_gp_variance = np.zeros((number_of_training_points, output_dimension))
        rollout_gp = np.zeros((number_of_training_points, output_dimension))
    
        new_state= copy.deepcopy(init_state)
        for t in range(number_of_training_points):
            #print('time:',t)
            state= np.array(np.concatenate((new_state,action_traj[t])))
            mean_set= np.zeros((train_y.shape[1],))
            variance_set = np.zeros((train_y.shape[1],))
            for k in range(train_y.shape[1]):
                K_star = self.get_pce_kernel_fast(train_x ,state.reshape((1,-1)),k)
                mean = K_star.T @ inv_val[k,:,:] @ train_y[:, k].reshape((-1,1))
                K_star_star = self.get_pce_kernel_fast(state.reshape((1, -1)), state.reshape((1, -1)), k)
    
                variance = K_star_star - K_star.T @ inv_val[k,:,:] @ K_star
                new_state[k] += mean
                #print(mean)
                mean_set[k] = mean
                variance_set[k] = variance
            pred_gp_mean[t] = mean_set
            pred_gp_variance[t] = variance_set
            rollout_gp[t] = new_state.copy()
    
        #print(train_y)
        #x=input()
        return pred_gp_mean,pred_gp_variance,rollout_gp




