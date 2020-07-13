# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:50:41 2020

@author: daksh
"""
from Data_interpret.GroundTruth import ReadMat
from gaussing_non_parametric import GP
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from Robofly_simulator.robot import Robot
from sklearn.preprocessing import normalize
import sys
if __name__ == '__main__':
    
    #Define training paramters
    
    TRAJECTORY_LENGTH=1 #seconds
    DELTA_T= 1/100
    NUM_TRAINING_EPOCHS=7 
    
    #Loading DATA
    project_folder = Path( "E:/Dropbox/Daksh/System_ID_project/system_identificatification_robofly/")# windows path
    data_folder = Path("Raw_data/8_8_2019 to 8_12_2019/")
    train_file = project_folder / data_folder / "2019-08-08-19-16-02_5sec.mat"
    train_file2=  project_folder / data_folder / "2019-08-09-13-01-42_10sec.mat"
    train_data = ReadMat(train_file)
    #train_data = ReadMat(train_file2)

    test_file= project_folder / data_folder / "2019-08-09-12-49-56_7sec.mat"
    test_data = ReadMat(test_file)
    
    #SAMPLING DATA
    sample_size= 100 # sampling of raw data
    train_state_traj , train_action_traj = train_data.sampled_mocap_data(sample_size) # state X time, number of actions X time
    test_state_traj , test_action_traj = test_data.sampled_mocap_data(sample_size)

    ### To start the trajectory after 0.4 seconds
    
    # train_state_traj = train_state_traj[:,40:]
    # train_action_traj = train_action_traj[:,40:]
    
    test_state_traj = test_state_traj[:,40:]
    test_action_traj = test_action_traj[:,40:]
    
    
    #converting to train and test
    train_x = np.concatenate((train_state_traj[:,:-1],train_action_traj[:,:-1]),axis=0)
    train_y = train_state_traj[:,1:]- train_state_traj[:,:-1]
    


    #choosing the prediction algorithm
    prediction_algo = 'blackbox_GP'
    #prediction_algo = 'parametric_model'
    #prediction_algo = 'parametric_model_with_GP'
#------------------for running  block box GP
    if prediction_algo=='blackbox_GP':
        #print(train_x.shape)
        # for i in range(12):
        #     plt.figure()
        #     plt.plot(train_x[i,:])
        # #normalize data
        #train_x= normalize(train_x.T)
        #train_x=train_x.T
        #print(train_x.shape)
        # for i in range(12):
        #     plt.figure()
        #     plt.plot(train_x[i,:])
        #plt.show()
        #sys.exit()
        #test_action_traj= normalize((test_action_traj.T))
        #test_action_traj= test_action_traj.T
        #----------------------------------------
        GP_obj= GP(DELTA_T,TRAJECTORY_LENGTH,NUM_TRAINING_EPOCHS)
        prediction_func= GP_obj.predict_gp
        #calculating inverse of kernel matrix
        print('start_inv')
        inv_val= GP_obj.get_inv_val(train_x.T,train_y.T,GP_obj.get_pce_kernel_fast)
        print('finish_inv')
        for epoch in range(NUM_TRAINING_EPOCHS):
            print('epoch # ',epoch)
            start_point = int(epoch*TRAJECTORY_LENGTH/DELTA_T)
            end_point = int((epoch+1)*TRAJECTORY_LENGTH/DELTA_T)
            init_state = test_state_traj[:,start_point]
            #init_state = train_state_traj[:,start_point]
            cur_action_traj =test_action_traj[:,start_point:end_point]
            print('start_point',start_point,'end_point', end_point)
            #cur_action_traj =train_action_traj[:,start_point:end_point]
            
            if epoch ==0:
                mean_trajs,variance_traj,rollout_trajs= prediction_func(train_x.T, train_y.T, init_state, cur_action_traj.T, inv_val)
                #GP_obj.update_hyperparams(train_y.T)
                #plt.plot(mean_trajs)
                #plt.show()
                #sys.exit()
            else:
                mean,variance,rollout= prediction_func(train_x.T, train_y.T, init_state, cur_action_traj.T, inv_val)
                mean_trajs= np.concatenate((mean_trajs,mean),axis=0)
                variance_traj= np.concatenate((variance_traj,variance),axis=0)
                rollout_trajs= np.concatenate((rollout_trajs,rollout),axis=0)
                #GP_obj.update_hyperparams(train_y.T)
    
#-------------------for running parametric
    elif prediction_algo=='parametric':
        dynamics_model= Robot()
        for epoch in range(NUM_TRAINING_EPOCHS):
            print('epoch # ',epoch)
            
            start_point = int(epoch*TRAJECTORY_LENGTH/DELTA_T)
            end_point = int((epoch+1)*TRAJECTORY_LENGTH/DELTA_T)
            init_state = test_state_traj[:,start_point]
            cur_action_traj =test_action_traj[:,start_point:end_point]
            time_interval = np.arange(0,TRAJECTORY_LENGTH,DELTA_T)
            assert time_interval.shape[0]==cur_action_traj.shape[1]
            if epoch ==0:
                rollout_trajs= dynamics_model.equation_simulate(init_state, time_interval, cur_action_traj)
                print(rollout_trajs.shape)
                sys.exit()
            else:
                rollout= dynamics_model.equation_simulate(init_state, time_interval, cur_action_traj)
                rollout_trajs= np.concatenate((rollout_trajs,rollout),axis=0)
            
        
    save_data={"rollouts":rollout_trajs, "state_traj":test_state_traj, "DELTA_T": DELTA_T,\
                "traj_length": TRAJECTORY_LENGTH,"num_epochs": NUM_TRAINING_EPOCHS}
    np.save('save_data.npy',save_data)
