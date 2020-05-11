import numpy as np
import matplotlib.pyplot as plt



if __name__== "__main__":
    rollout_trajs= np.load('E:\Insect Robo Lab\Dropbox\Dropbox\Daksh\system_identificatification_robofly/rollout_trajs.npy')
    state_traj= np.load('E:\Insect Robo Lab\Dropbox\Dropbox\Daksh\system_identificatification_robofly/state_trajs.npy')
    rollout_trajs=rollout_trajs.T
    state_traj= state_traj[:,:rollout_trajs.shape[1]]
    X= np.linspace(0,state_traj.shape[1]/100,state_traj.shape[1] )
    fig,a =  plt.subplots(3,2)
    print(X.shape, rollout_trajs.shape)
    a[0][0].plot(X, state_traj[0,:])
    a[0][0].plot(X, rollout_trajs[0,:])
    a[0][0].set_title('X Position vs time')
    #a[0][0].legend(('Xposition', 'Yposition', 'Zposition'))
    a[0][1].plot(X, state_traj[1,:])
    a[0][1].plot(X, rollout_trajs[1,:])
    a[0][1].set_title('Y Position vs time')

    a[1][0].plot(X, state_traj[2,:])
    a[1][0].plot(X, rollout_trajs[2,:])
    a[1][0].set_title('Z Position vs time')

    a[1][1].plot(X, state_traj[3,:])
    a[1][1].plot(X, rollout_trajs[3,:])
    a[1][1].set_title('Roll vs time')

    a[2][0].plot(X, state_traj[4,:])
    a[2][0].plot(X, rollout_trajs[4,:])
    a[2][0].set_title('Pitch vs time')

    plt.show()
