import numpy as np
import matplotlib.pyplot as plt



if __name__== "__main__":
    path= 'E:\Insect Robo Lab\Dropbox\Dropbox\Daksh\system_identificatification_robofly/save_data.npy'
    raw_data= np.load(path, allow_pickle=True)
    state_traj = raw_data.item().get('state_traj')
    rollout_trajs = raw_data.item().get('rollouts').T
    trajectory_size = raw_data.item().get('traj_length')
    DELTA_T = raw_data.item().get('DELTA_T')
    total_points = rollout_trajs.shape[1]

    state_traj= state_traj[:,:rollout_trajs.shape[1]]
    X= np.linspace(0,state_traj.shape[1]*DELTA_T,state_traj.shape[1] )
    fig,a =  plt.subplots(3,2)
    prev=0
    for i in range(int(trajectory_size/DELTA_T),total_points,int(trajectory_size/DELTA_T)):
        a[0][0].plot(X[prev:i], state_traj[0,prev:i],'b')
        a[0][0].plot(X[prev:i], rollout_trajs[0,prev:i],'g')
        a[0][0].set_title('X Position vs time')
        #a[0][0].legend(('Xposition', 'Yposition', 'Zposition'))
        a[0][1].plot(X[prev:i], state_traj[1,prev:i],'b')
        a[0][1].plot(X[prev:i], rollout_trajs[1,prev:i],'g')
        a[0][1].set_title('Y Position vs time')

        a[1][0].plot(X[prev:i], state_traj[2,prev:i],'b')
        a[1][0].plot(X[prev:i], rollout_trajs[2,prev:i],'g')
        a[1][0].set_title('Z Position vs time')

        a[1][1].plot(X[prev:i], state_traj[6,prev:i],'b')
        a[1][1].plot(X[prev:i], rollout_trajs[6,prev:i],'g')
        a[1][1].set_title('Roll vs time')

        a[2][0].plot(X[prev:i], state_traj[7,prev:i],'b')
        a[2][0].plot(X[prev:i], rollout_trajs[7,prev:i],'g')
        a[2][0].set_title('Pitch vs time')
        prev=i
    fig,b =  plt.subplots(3,2)
    prev=0
    for i in range(int(trajectory_size/DELTA_T),total_points,int(trajectory_size/DELTA_T)):
        b[0][0].plot(X[prev:i], state_traj[3,prev:i],'b')
        b[0][0].plot(X[prev:i], rollout_trajs[3,prev:i],'g')
        b[0][0].set_title('X Velocfity vs time')
        #a[0][0].legend(('Xposition', 'Yposition', 'Zposition'))
        b[0][1].plot(X[prev:i], state_traj[4,prev:i],'b')
        b[0][1].plot(X[prev:i], rollout_trajs[4,prev:i],'g')
        b[0][1].set_title('Y Velocity vs time')

        b[1][0].plot(X[prev:i], state_traj[5,prev:i],'b')
        b[1][0].plot(X[prev:i], rollout_trajs[5,prev:i],'g')
        b[1][0].set_title('Z Velocity vs time')

        b[1][1].plot(X[prev:i], state_traj[9,prev:i],'b')
        b[1][1].plot(X[prev:i], rollout_trajs[9,prev:i],'g')
        b[1][1].set_title('Ang vel rol vs time')

        b[2][0].plot(X[prev:i], state_traj[10,prev:i],'b')
        b[2][0].plot(X[prev:i], rollout_trajs[10,prev:i],'g')
        b[2][0].set_title('Ang vel Pith vs time')
        prev=i

    plt.show()
