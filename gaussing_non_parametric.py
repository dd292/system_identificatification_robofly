from Data_interpret.GroundTruth import ReadMat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy
from scipy import sparse
from Data_interpret.visualization import TrajectoryVisualize

DELTA_T =1/100
TRAJECTORY_LENGTH = 1#seconds
NUM_TRAINING_EPOCHS= 5
kernel_length_scales = np.ones((15,12))*1
kernel_scale_factors = np.ones((12,1))*0.5
noise_sigmas = np.array([.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05])*0
def make_training_data(epoch, state_traj, action_traj):
    start_point= int(epoch*TRAJECTORY_LENGTH/DELTA_T)
    end_point = int((epoch+1)*TRAJECTORY_LENGTH/DELTA_T)
    cur_state_traj= state_traj[:,start_point:end_point]
    cur_action_traj= action_traj[:,start_point:end_point]
    delta_state_traj= cur_state_traj[:,1:]- cur_state_traj[:,:-1]
    x= np.concatenate((cur_state_traj[:,:-1],cur_action_traj[:,:-1]),axis=0)
    return x,delta_state_traj,cur_action_traj
def get_pce_kernel_fast(X, X_prime, Gp_number):


    K= np.zeros((X.shape[0],X_prime.shape[0]))
    sigma_f= kernel_scale_factors[Gp_number]
    l= kernel_length_scales[:, Gp_number];
    for iter1,i in enumerate(X):
        K [iter1,:] = sigma_f*np.exp(-np.sum((i-X_prime)**2/(2*l**2),axis=1))
    #print(K.shape)

    return K
def get_inv_val(train_x,train_y,func):
    inv_value = np.zeros((train_y.shape[1],train_x.shape[0],train_x.shape[0]))
    for k in range(train_y.shape[1]):
        inv_value[k,:] = np.linalg.inv(func(train_x,train_x,k)+ noise_sigmas[k]**2*np.eye(train_x.shape[0]))
    return inv_value

def predict_gp(train_x,train_y,init_state, action_traj,inv_val):

    output_dimension= train_y.shape[1]

    number_of_training_points=int(TRAJECTORY_LENGTH/DELTA_T)
    pred_gp_mean = np.zeros((number_of_training_points, output_dimension))
    pred_gp_variance = np.zeros((number_of_training_points, output_dimension))
    rollout_gp = np.zeros((number_of_training_points, output_dimension))

    new_state= copy.deepcopy(init_state)
    for t in range(number_of_training_points):
        print('time:',t)
        state= np.array(np.concatenate((new_state,action_traj[t])))
        mean_set= np.zeros((train_y.shape[1],))
        variance_set = np.zeros((train_y.shape[1],))
        for k in range(train_y.shape[1]):
            K_star = get_pce_kernel_fast(train_x ,state.reshape((1,-1)),k)
            mean = K_star.T @ inv_val[k,:,:] @ train_y[:, k].reshape((-1,1))
            K_star_star = get_pce_kernel_fast(state.reshape((1, -1)), state.reshape((1, -1)), k)

            variance = K_star_star - K_star.T @ inv_val[k,:,:] @ K_star
            new_state[k] += mean
            mean_set[k] = mean
            variance_set[k] = variance
        pred_gp_mean[t] = mean_set
        pred_gp_variance[t] = variance_set
        rollout_gp[t] = new_state.copy()

    #print(train_y)
    #x=input()
    return pred_gp_mean,pred_gp_variance,rollout_gp





if __name__ == '__main__':
    #project_folder = Path("/media/daksh/52BCB7EEBCB7CAAD/Insect Robo Lab/Dropbox/Dropbox/Daksh/system_identificatification_robofly/") # linux path
    #LOading DATA
    project_folder = Path( "E:/Insect Robo Lab/Dropbox/Dropbox/Daksh/system_identificatification_robofly/")# windows path
    data_folder = Path("Raw_data/8_8_2019 to 8_12_2019/")
    train_file = project_folder / data_folder / "2019-08-08-19-16-02_5sec.mat"
    train_data = ReadMat(train_file)

    test_file= project_folder / data_folder / "2019-08-09-12-49-56_7sec.mat"
    test_data = ReadMat(test_file)

    #SAMPLING DATA
    sample_size= 100 # sampling of raw data
    train_state_traj , train_action_traj = train_data.sampled_mocap_data(sample_size) # state X time, number of actions X time

    test_state_traj , test_action_traj = test_data.sampled_mocap_data(sample_size)
    #converting to train and test
    train_x = np.concatenate((train_state_traj[:,:-1],train_action_traj[:,:-1]),axis=0)
    train_y = train_state_traj[:,1:]-train_state_traj[:,:-1]

    #calculating inverse of kernel matrix

    print('start_inv')
    inv_val= get_inv_val(train_x.T,train_y.T,get_pce_kernel_fast)
    print('finish_inv')

    for epoch in range(NUM_TRAINING_EPOCHS):

        start_point = int(epoch*TRAJECTORY_LENGTH/DELTA_T)
        end_point = int((epoch+1)*TRAJECTORY_LENGTH/DELTA_T)
        #init_state = test_state_traj[:,start_point]
        init_state = train_state_traj[:,start_point]
        #cur_action_traj =test_action_traj[:,start_point:end_point]
        cur_action_traj =train_action_traj[:,start_point:end_point]
        if epoch ==0:
            mean_trajs,variance_traj,rollout_trajs= predict_gp(train_x.T, train_y.T, init_state, cur_action_traj.T, inv_val)
        else:
            mean,variance,rollout= predict_gp(train_x.T, train_y.T, init_state, cur_action_traj.T, inv_val)
            mean_trajs= np.concatenate((mean_trajs,mean),axis=0)
            variance_traj= np.concatenate((variance_traj,variance),axis=0)
            rollout_trajs= np.concatenate((rollout_trajs,rollout),axis=0)

    save_data={"rollouts":rollout_trajs, "state_traj":train_state_traj, "DELTA_T": DELTA_T,\
                "traj_length": TRAJECTORY_LENGTH,"num_epochs": NUM_TRAINING_EPOCHS,\
               "kernel_length_scales": kernel_length_scales,\
               "kernel_scale_factors":kernel_scale_factors,\
                "noise_sigmas":noise_sigmas}
    np.save('save_data.npy',save_data)








def training_epochs():
## For training epochs
    project_folder = Path( "E:/Insect Robo Lab/Dropbox/Dropbox/Daksh/system_identificatification_robofly/")# windows path
    data_folder = Path("Raw_data/8_8_2019 to 8_12_2019/")
    file_to_open = project_folder / data_folder / "2019-08-08-19-16-02_5sec.mat"
    file = ReadMat(file_to_open)
    sample_size= 100 # sampling of raw data
    state_traj, action_traj = file.sampled_mocap_data(sample_size) # state X time, number of actions X time

    for epoch in range(NUM_TRAINING_EPOCHS):


        start_point = int(epoch*TRAJECTORY_LENGTH/DELTA_T)
        init_state = state_traj[:,start_point]


        if epoch==0:
            train_x, train_y, cur_action_traj = make_training_data(epoch, state_traj, action_traj)
            mean_trajs, variance_traj, rollout_trajs = predict_gp(train_x.T,train_y.T,init_state, cur_action_traj.T)
        else:
            new_train_x, new_train_y,cur_action_traj = make_training_data(epoch, state_traj, action_traj)
            train_x = np.concatenate((train_x, new_train_x),axis=1)
            train_y = np.concatenate((train_y, new_train_y),axis=1)
            mean, variance, rollout = predict_gp(train_x.T,train_y.T,init_state, cur_action_traj.T)
            mean_trajs= np.concatenate((mean_trajs,mean),axis=0)
            variance_traj= np.concatenate((variance_traj,variance),axis=0)
            rollout_trajs= np.concatenate((rollout_trajs,rollout),axis=0)




