import numpy as np
from scipy.io import loadmat
from scipy import signal
from pathlib import Path
import math
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
        return np.array([xpos, ypos, zpos, xvel,yvel,zvel, roll, pitch, yaw, omega1,omega2,omega3])

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


if __name__ == '__main__':
    
    project_folder = Path( "E:/Dropbox/Daksh/System_ID_project/system_identificatification_robofly/")# windows path
    data_folder = Path("Raw_data/8_8_2019 to 8_12_2019/")
    file_to_open = project_folder / data_folder / "2019-08-09-13-01-42_10sec.mat"
    object = ReadMat(file_to_open)
    
    
    
    sampling= 100
    state_traj, action_traj= object.sampled_mocap_data(sampling)# state->[body_vel,body_angles,body_angular_velocities]
    state_traj= state_traj[:,:900]
    Y = np.linspace(0, state_traj.shape[1]/100, state_traj.shape[1])
    accelerations= np.gradient(state_traj,0.01)#(state_traj[:,1:]- state_traj[:,:-1])/0.01
    accelerations= accelerations[1]
    print(accelerations.shape)
    
    
    #Y = np.linspace(0, accelerations.shape[1]/100, accelerations.shape[1])

    import matplotlib.pyplot as plt
    fig,a =  plt.subplots(2,2)
    
    a[0][0].plot(Y, accelerations[0,:])
    a[0][0].plot(Y, accelerations[1,:])
    a[0][0].plot(Y, accelerations[2,:])
    a[0][0].set_title('X,Y,Z vel vs time ')
    a[0][0].legend(('X_vel', 'Y_vel', 'Z_vel'))

    a[1][0].plot(Y, accelerations[3,:])
    a[1][0].plot(Y, accelerations[4,:])
    a[1][0].plot(Y, accelerations[5,:])
    a[1][0].set_title('X,Y,Z accelerations vs time')
    a[1][0].legend(('X_acc', 'Y_acc', 'Z_acc'))
    
    a[0][1].plot(Y, accelerations[6,:])
    a[0][1].plot(Y, accelerations[7,:])
    a[0][1].plot(Y, accelerations[8,:])
    a[0][1].set_title('angular vel vs time')
    a[0][1].legend(('theta_X_vel', 'theta_Y_vel', 'theta_Z_vel'))

    a[1][1].plot(Y, accelerations[9,:])
    a[1][1].plot(Y, accelerations[10,:])
    a[1][1].plot(Y, accelerations[11,:])
    a[1][1].set_title('angular accelerations vs time')
    a[1][1].legend(('theta_X_acc', 'theta_Y_acc', 'theta_Z_acc'))
    plt.show()


