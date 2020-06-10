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
import sys

if __name__ == '__main__':

    # Define training paramters

    TRAJECTORY_LENGTH = 1  # seconds
    DELTA_T = 1 / 100
    NUM_TRAINING_EPOCHS = 7

    # Loading DATA
    project_folder = Path("E:/Dropbox/Daksh/System_ID_project/system_identificatification_robofly/")  # windows path
    data_folder = Path("Raw_data/8_8_2019 to 8_12_2019/")
    train_file = project_folder / data_folder / "2019-08-08-19-16-02_5sec.mat"
    train_file2 = project_folder / data_folder / "2019-08-09-13-01-42_10sec.mat"
    train_data = ReadMat(train_file)
    # train_data = ReadMat(train_file2)

    test_file = project_folder / data_folder / "2019-08-09-12-49-56_7sec.mat"
    test_data = ReadMat(test_file)

    # SAMPLING DATA
    sample_size = 100  # sampling of raw data
    train_state_traj, train_action_traj = train_data.sampled_mocap_data_dynamics_world(sample_size)  # state X time, number of actions X time
    test_state_traj, test_action_traj = test_data.sampled_mocap_data_dynamics_world(sample_size)
    print(max(train_action_traj[0, :]))
    print(max(train_action_traj[1, :]))
    print(max(train_action_traj[2, :]))
    print(min(train_action_traj[0, :]))
    print(min(train_action_traj[1, :]))
    print(min(train_action_traj[2, :]))
    sys.exit()
    ### To start the trajectory after 0.4 seconds

    # train_state_traj = train_state_traj[:,40:]
    # train_action_traj = train_action_traj[:,40:]

    test_state_traj = test_state_traj[:, 40:]
    test_action_traj = test_action_traj[:, 40:]

    # converting to train and test
    train_x = np.concatenate((train_state_traj[:, :-1], train_action_traj[:, :-1]), axis=0)
    train_y = train_state_traj[:, 1:] - train_state_traj[:, :-1]
    dynamics_model = Robot()
    for epoch in range(NUM_TRAINING_EPOCHS):
        print('epoch # ', epoch)

        start_point = int(epoch * TRAJECTORY_LENGTH / DELTA_T)
        end_point = int((epoch + 1) * TRAJECTORY_LENGTH / DELTA_T)
        init_state = test_state_traj[:, start_point]
        cur_action_traj = test_action_traj[:, start_point:end_point]
        time_interval = np.arange(0, TRAJECTORY_LENGTH, DELTA_T)
        assert time_interval.shape[0] == cur_action_traj.shape[1]
        if epoch == 0:
            rollout_trajs = dynamics_model.equation_simulate(init_state, time_interval, cur_action_traj)
        else:
            rollout = dynamics_model.equation_simulate(init_state, time_interval, cur_action_traj)
            rollout_trajs = np.concatenate((rollout_trajs, rollout), axis=0)

    save_data = {"rollouts": rollout_trajs, "state_traj": test_state_traj, "DELTA_T": DELTA_T, \
                 "traj_length": TRAJECTORY_LENGTH, "num_epochs": NUM_TRAINING_EPOCHS}
    np.save('save_data.npy', save_data)
