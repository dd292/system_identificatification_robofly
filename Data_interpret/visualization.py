import matplotlib.pyplot as plt
import numpy as np
class TrajectoryVisualize():
    def __init__(self):
       self.figure_number=0

    def plot_angles(self,roll,pitch,yaw):
        X = np.linspace(0, roll.shape[0] / 10000, roll.shape[0])
        plt.plot(X, roll)
        plt.plot(X, pitch)
        #plt.plot(X, yaw)
        plt.legend(('Roll', 'Pitch'))#, 'Yaw'))
        #plt.ylim(top=1, bottom=-1)


    def plot_positions(self, xpos, ypos, zpos):
        X = np.linspace(0, xpos.shape[0] / 10000, xpos.shape[0])
        plt.plot(X, xpos)
        plt.plot(X, ypos)
        plt.plot(X, zpos)
        plt.legend(('Roll_acc', 'pitch_acc', 'Z'))
        #plt.ylim(top=1, bottom=-1)


    def plot_vel(self, xvel, yvel, zvel):
        X = np.linspace(0, zvel.shape[0] / 10000, zvel.shape[0])
        plt.plot(X, xvel)
        plt.plot(X, yvel)
        plt.plot(X, zvel)
        plt.legend(('X_velocity', 'Y_velocity', 'Z_velocity'))
        #plt.ylim(top=1, bottom=-1)



    def plotter(self, dict):
        for key,value in dict.items():
            plt.figure(self.figure_number)
            if key=='angles':
                self.plot_angles(value[0],value[1],value[2])
            elif key=='pos':
                self.plot_positions(value[0],value[1],value[2])
            elif key == 'vel':
                self.plot_vel(value[0], value[1], value[2])
            self.figure_number+=1

    def show_plot(self):
        plt.show()
