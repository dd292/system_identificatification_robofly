import numpy as np
from scipy.io import loadmat
from pathlib import Path
import math
from visualization import TrajectcoryVisualize
from scipy.misc import derivative
# Script is written by Daksh Dhingra, 2020
# input ->.mat file containingthe motioncapture data
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
# |action            |   maximum Value   |   minimum Value   |
#  -----------------------------------------------------
# |Z acceleration    |   inf             |   -inf            |
# |Roll Acceleration |   inf             |   -inf            |
# |Pitch Acceleration|   inf             |   -inf            |
#
# Convention:
# Rotation about body X= Roll
# Rotation about body Y= Pitch
class ReadMat:
    def __init__(self,filename):
        self.input_file = filename
        self.raw_data=loadmat(self.input_file)['yout']


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
    def pos_vel(self,xpos,ypos,zpos):
        xvel= derivative(xpos)
        yvel= derivative(ypos)
        zvel= derivative(zpos)
        print(xvel[0:200])
        return xvel,yvel,zvel
    #def velocities(self,positions,angles):


    #def get_states(self):

    #def get_actions(self):
if __name__ == '__main__':
    project_folder = Path("/media/daksh/52BCB7EEBCB7CAAD/Insect Robo Lab/Dropbox/Dropbox/Daksh/system_identificatification_robofly/")
    data_folder = Path("Raw_data/8_8_2019 to 8_12_2019/")
    file_to_open = project_folder / data_folder / "2019-08-08-19-16-02_5sec.mat"
    object = ReadMat(file_to_open)
    roll, pitch, yaw = object.angles()
    xpos, ypos, zpos= object.positions()
    xvel, yvel, zvel = object.pos_vel(xpos, ypos, zpos)
    vis= TrajectcoryVisualize()
    vis.plotter({"angles": (roll,pitch,yaw), "pos":(xpos,ypos,zpos), "vel": (xvel, yvel, zvel)})