import numpy as np
from scipy.io import loadmat
from scipy import signal
from pathlib import Path
import math
import scipy.io as sio
#from visualization import TrajectoryVisualize
from scipy.misc import derivative
import matplotlib.pyplot as plt
# Script is written by Daksh Dhingra, 2020
# input ->.mat file containing the motioncapture data
# output :
# state_trajetory
# action_trajectory


# state_limits
# |state        |   maximum Value   |   minimum Value   |
#  -----------------------------------------------------
# |X position   |   inf             |   -inf            |
# |Y position   |   inf             |   -inf            |
# |Z position   |   inf             |   -inf            |
# |X velocity   |   inf             |   -inf            |
# |Y velocity   |   inf             |   -inf            |
# |Z velocity   |   inf             |   -inf            |
# |theta_x_body |   inf             |   -inf            |
# |theta_y_body |   inf             |   -inf            |
# |theta_z_body |   inf             |   -inf            |
# |omega_x_world|   inf             |   -inf            |
# |omega_y_world|   inf             |   -inf            |
# |omega_z_world|   inf             |   -inf            |
#  -----------------------------------------------------
# Action Limits
# |action       |   maximum Value   |   minimum Value   |
#  -----------------------------------------------------
# |Z command    |   inf             |   -inf            |
# |Roll command |   inf             |   -inf            |
# |Pitch command|   inf             |   -inf            |
#
# Convention:
# Rotation about body X= Roll
# Rotation about body Y= Pitch
class ReadMat:
    def __init__(self,filename):
        self.input_file = filename
        self.all_data= loadmat(self.input_file)

        self.raw_data= self.all_data['yout']
        self.params= self.all_data['tgParams']




    def positions(self):
        return np.array([self.raw_data[:,10], self.raw_data[:,11], self.raw_data[:,12]])

    def angles(self):
        q0 = np.array(self.raw_data[:,13])
        q1 = np.array(self.raw_data[:,14])
        q2 = np.array(self.raw_data[:,15])
        q3 = np.array(self.raw_data[:,16])
        roll= np.zeros((q0.shape))
        pitch = np.zeros((q0.shape))
        yaw = np.zeros((q0.shape))
        a=0
        for i,j,k,l in zip(q0,q1,q2,q3):

            roll[a]= math.atan2 ( 2*(i*j + k*l), 1-2*(j**2 + k**2))
            val= 2 * (i * k - l * j)
            if abs(val)>=1:
                pitch[a]= math.copysign(math.pi/2,val)
            else:
                pitch[a] = math.asin(val)
            yaw[a] = math.atan2 (2*(i*l + j*k ),1-2*(k**2+l**2))
            a+=1
        return roll, pitch, yaw
    def get_rotation_matrix(self):
        R= np.array(self.raw_data[:,20:29])
        R= R.reshape((-1,3,3))
        return R

    def vel(self,xpos,ypos,zpos):
        sampling_rate= xpos.shape[0]
        cut_off_frequency= 240/10*2*np.pi/(sampling_rate/2) #0.054 HZ
        b,a= signal.butter (3, cut_off_frequency) #third order butterworth filter with cutoff frequency given above
        filtered_xpos =signal.filtfilt(b,a,xpos)
        filtered_ypos =signal.filtfilt(b,a,ypos)
        filtered_zpos =signal.filtfilt(b,a,zpos)
        xvel= np.gradient(filtered_xpos,1/10000)
        yvel= np.gradient(filtered_ypos,1/10000)
        zvel= np.gradient(filtered_zpos,1/10000)
        R= self.get_rotation_matrix()
        X_vel_body = np.zeros((xvel.shape))
        Y_vel_body = np.zeros((xvel.shape))
        Z_vel_body = np.zeros((xvel.shape))
        
        for (iter1,(i,j,k,rot)) in enumerate(zip(xvel,yvel,zvel,R)):
            vel_body= rot.T@ np.array([i,j,k])
            X_vel_body[iter1]=vel_body[0]
            Y_vel_body[iter1]=vel_body[1]
            Z_vel_body[iter1]=vel_body[2]
            
        omega1= self.raw_data[:,17]
        omega2= self.raw_data[:,18]
        omega3= self.raw_data[:,19]
        # print(xvel[0:200])

        return X_vel_body, Y_vel_body, Z_vel_body, omega1, omega2, omega3

    def get_states(self):
        roll, pitch, yaw = self.angles()
        xpos, ypos, zpos= self.positions()
        xvel,yvel,zvel, omega1,omega2,omega3 = self.vel(xpos, ypos, zpos)
        return np.array([xvel,yvel,zvel, roll, pitch, yaw, omega1,omega2,omega3])

    def get_states_world_dynamics(self):
        roll, pitch, yaw = self.angles()
        xpos, ypos, zpos= self.positions()
        xvel,yvel,zvel, omega1,omega2,omega3 = self.vel(xpos, ypos, zpos)
        return np.array([roll,pitch,yaw,omega1, omega2, omega3, xpos ,ypos,zpos, xvel,yvel,zvel])
    def get_actions(self):
        initial_amplitude= (self.params[0][0][5][0][0][2]).item()
        Z_param= self.raw_data[:,6]-initial_amplitude
        roll_param= self.raw_data[:,7]
        pitch_param= self.raw_data[:,8]
        return np.array([Z_param, roll_param, pitch_param])
        
        
    def sampled_mocap_data(self, samples_steps):
        states_trajectory= self.get_states() # number of state X number of training examples
        action_trajectory= self.get_actions() # number of actions X number of training examples
        new_action_trajectory= np.zeros((action_trajectory.shape[0],int(action_trajectory.shape[1]/samples_steps)))
        new_trajectory= np.zeros((states_trajectory.shape[0],int(states_trajectory.shape[1]/samples_steps)))
        for i in range(int(states_trajectory.shape[1]/samples_steps)):
            new_trajectory[:,i] =np.sum(states_trajectory[:,i*samples_steps:(i+1)*samples_steps],axis=1)/samples_steps
            new_action_trajectory[:,i] =np.sum(action_trajectory[:,i*samples_steps:(i+1)*samples_steps],axis=1)/samples_steps

        return new_trajectory,new_action_trajectory

    def sampled_mocap_data_dynamics_world(self, samples_steps):
        states_trajectory= self.get_states_world_dynamics() # number of state X number of training examples
        action_trajectory= self.get_actions() # number of actions X number of training examples
        new_action_trajectory= np.zeros((action_trajectory.shape[0],int(action_trajectory.shape[1]/samples_steps)))
        new_trajectory= np.zeros((states_trajectory.shape[0],int(states_trajectory.shape[1]/samples_steps)))
        for i in range(int(states_trajectory.shape[1]/samples_steps)):
            new_trajectory[:,i] =np.sum(states_trajectory[:,i*samples_steps:(i+1)*samples_steps],axis=1)/samples_steps
            new_action_trajectory[:,i] =np.sum(action_trajectory[:,i*samples_steps:(i+1)*samples_steps],axis=1)/samples_steps

        return new_trajectory,new_action_trajectory


if __name__ == '__main__':
    # Loading DATA
    project_folder = Path("E:/Dropbox/Daksh/System_ID_project/system_identificatification_robofly/")  # windows path
    data_folder = Path("Raw_data/8_8_2019 to 8_12_2019/")
    train_file = project_folder / data_folder / "2019-08-08-19-16-02_5sec.mat"
    train_file2 = project_folder / data_folder / "2019-08-09-13-01-42_10sec.mat"
    train_data = ReadMat(train_file)
    train_data2 = ReadMat(train_file2)
    # train_data = ReadMat(train_file2)

    test_file = project_folder / data_folder / "2019-08-09-12-49-56_7sec.mat"
    test_data = ReadMat(test_file)

    # SAMPLING DATA
    sample_size = 100  # sampling of raw data
    train_state_traj, train_action_traj = train_data.sampled_mocap_data(sample_size)  # state X time, number of actions X time
    train_state_traj2, train_action_traj2 = train_data2.sampled_mocap_data(sample_size)  # state X time, number of actions X time
    test_state_traj, test_action_traj = test_data.sampled_mocap_data(sample_size)
    sio.savemat('traj1.mat',{'states':train_state_traj,'actions': train_action_traj})
    sio.savemat('traj2.mat', {'states': train_state_traj2, 'actions': train_action_traj2})
    sio.savemat('traj3.mat', {'states': test_state_traj, 'actions': test_action_traj})

    #sampling= 100
    #state_traj, action_traj= object.sampled_mocap_data(sampling)# state->[body_vel,body_angles,body_angular_velocities]
    #
    #roll, pitch, yaw = object.angles()
    #xpos, ypos, zpos= object.positions()
    #xvel,yvel,zvel, omega1,omega2,omega3 = object.vel(xpos,ypos,zpos)
    #Z_param, roll_param, pitch_param= object.get_actions()
    #vis= TrajectoryVisualize()
    
    #vis.plotter({"pos": (state_traj[0,:],state_traj[1,:],state_traj[2,:]),"angles": (state_traj[3,:],state_traj[4,:],state_traj[5,:]),"vel": (state_traj[6,:],state_traj[7,:],state_traj[8,:])})
    # vis.plotter({"vel": (Z_param, roll_param, pitch_param)})
    #vis.plotter({"pos": (action_traj[0,:],action_traj[1,:],action_traj[2,:])})
    #vis.show_plot()
    # import matplotlib.pyplot as plt
    # fig,a =  plt.subplots(2,2)

    # X = np.linspace(0, xpos.shape[0] / 10000, xpos.shape[0])
    # Y = np.linspace(0, state_traj.shape[1]/100, state_traj.shape[1])

    # a[0][0].plot(X, xpos)
    # a[0][0].plot(X, ypos)
    # a[0][0].plot(X, zpos)
    # a[0][0].set_title('Position vs time (N=55k)')
    # a[0][0].legend(('Xposition', 'Yposition', 'Zposition'))
    # a[0][1].plot(Y, state_traj[0,:])
    # a[0][1].plot(Y, state_traj[1,:])
    # a[0][1].plot(Y, state_traj[2,:])
    # a[0][1].set_title('Sampled Position vs time (N=553)')
    # a[0][1].legend(('Xposition', 'Yposition', 'Zposition'))

    # a[1][0].plot(X, xvel)
    # a[1][0].plot(X, yvel)
    # a[1][0].plot(X, zvel)
    # a[1][0].set_title('Velocity vs time (N=55k)')
    # a[1][0].legend(('Xvel', 'Yvel', 'Zvel'))
    # a[1][1].plot(Y, state_traj[3,:])
    # a[1][1].plot(Y, state_traj[4,:])
    # a[1][1].plot(Y, state_traj[5,:])
    # a[1][1].set_title('Sampled velocity vs time (N=553)')
    # a[1][1].legend(('Xvel', 'Yvel', 'Zvel'))

    # fig,b =  plt.subplots(2,2)
    # b[0][0].plot(X, roll)
    # b[0][0].plot(X, pitch)
    # b[0][0].set_title('Attitude vs time (N=55k)')
    # b[0][0].legend(('roll', 'Pitch'))
    # b[0][1].plot(Y, state_traj[6,:])
    # b[0][1].plot(Y, state_traj[7,:])
    # b[0][1].set_title('Sampled attitude vs time (N=553)')
    # b[0][1].legend(('roll', 'Pitch'))

    # b[1][0].plot(X, Z_param)
    # b[1][0].plot(X, roll_param)
    # b[1][0].plot(X, pitch_param)
    # b[1][0].set_title('actions vs time (N=55k)')
    # b[1][0].legend(('Z_acc', 'Roll_acc', 'Pitch_acc'))
    # b[1][1].plot(Y, action_traj[0,:])
    # b[1][1].plot(Y, action_traj[1,:])
    # b[1][1].plot(Y, action_traj[2,:])
    # b[1][1].set_title('Sampled actions vs time (N=553)')
    # b[1][1].legend(('Z_acc', 'Roll_acc', 'Pitch_acc'))
    # plt.show()

