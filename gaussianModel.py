import numpy as np
import sys
import copy
# Global variables
NUM_TRAINING_EPOCHS = 12
NUM_DATAPOINTS_PER_EPOCH = 50
NUM_TRAJ_SAMPLES = 10
DELTA_T = 0.05
rng = np.random.RandomState(12345)

# State representation
# dtheta, dx, theta, x

kernel_length_scales = np.array([[240.507, 242.9594, 218.0256, 203.0197],
                                 [175.9314, 176.8396, 178.0185, 33.0219],
                                 [7.4687, 7.3903, 13.0914, 34.6307],
                                 [0.8433, 1.0499, 1.2963, 2.3903],
                                 [0.781, 0.9858, 1.7216, 31.2894],
                                 [23.1603, 24.6355, 49.9782, 219.185]])
kernel_scale_factors = np.array([3.5236, 1.3658, 0.7204, 1.1478])
noise_sigmas = np.array([0.0431, 0.0165, 0.0145, 0.0143])


def sim_rollout(sim, policy, n_steps, dt, init_state):
    """
    :param sim: the simulator
    :param policy: policy that generates rollout
    :param n_steps: number of time steps to run
    :param dt: simulation step size
    :param init_state: initial state
    :return: times:   a numpy array of size [n_steps + 1]
             states:  a numpy array of size [n_steps + 1 x 4]
             actions: a numpy array of size [n_steps]
                        actions[i] is applied to states[i] to generate states[i+1]
    """
    states = []
    state = init_state
    actions = []
    for i in range(n_steps):
        states.append(state)
        action = policy.predict(state)
        actions.append(action)
        state = sim.step(state, [action], noisy=True)

    states.append(state)
    times = np.arange(n_steps + 1) * dt
    return times, np.array(states), np.array(actions)


def augmented_state(state, action):
    """
    :param state: cartpole state
    :param action: action applied to state
    :return: an augmented state for training GP dynamics
    """
    dtheta, dx, theta, x = state
    return x, dx, dtheta, np.sin(theta), np.cos(theta), action


def make_training_data(state_traj, action_traj, delta_state_traj):
    """
    A helper function to generate training data.
    """
    x = np.array([augmented_state(state, action) for state, action in zip(state_traj, action_traj)])
    y = delta_state_traj
    return x, y

def get_pce_kernel_slow(X, X_prime, Gp_number):
    K= np.zeros((X.shape[0],X_prime.shape[0]))
    sigma_f= kernel_scale_factors[Gp_number]
    l= kernel_length_scales[:, Gp_number];
    for iter1,i in enumerate(X):
        for iter2,j in enumerate(X_prime):
            K [iter1,iter2] = sigma_f*np.exp(-np.sum((i-j)**2/(2*l**2)))
    #print(K.shape)
    return K
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

def predict_gp(train_x, train_y, init_state, action_traj):
    """
    Let M be the number of training examples
    Let H be the length of an epoch (NUM_DATAPOINTS_PER_EPOCH)
    Let N be the number of trajectories (NUM_TRAJ_SAMPLES)
    NOTE: Please use rng.normal(mu, sigma) to generate Gaussian random noise.
          https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.normal.html
    :param train_x: a numpy array of size [M x 6]
    :param train_y: a numpy array of size [M x 4]
    :param init_state: a numpy array of size [4]. Initial state of current epoch.
                       Use this to generate rollouts.
    :param action_traj: a numpy array of size [M]
    :return:
             # This is the mean rollout
             pred_gp_mean: a numpy array of size [H x 4]
                           This is mu_t[k] in Algorithm 1 in the HW1 PDF.
             pred_gp_variance: a numpy array of size [H x 4].
                               This is sigma_t[k] in Algorithm 1 in the HW1 PDF.
             rollout_gp: a numpy array of size [H x 4]
                         This is x_t[k] in Algorithm 1 in the HW1 PDF.
                         It should start from t=1, i.e. rollout_gp[0,k] = x_1[k]

             # These are the sampled rollouts
             pred_gp_mean_trajs: a numpy array of size [N x H x 4]
                                 This is mu_t^j[k] in Algorithm 2 in the HW1 PDF.
             pred_gp_variance_trajs: a numpy array of size [N x H x 4]
                                     This is sigma_t^j[k] in Algorithm 2 in the HW1 PDF.
             rollout_gp_trajs: a numpy array of size [N x H x 4]
                               This is x_t^j[k] in Algorithm 2 in the HW1 PDF.
                               It should start from t=1, i.e. rollout_gp_trajs[j,0,k] = x_1^j[k]
    """

    # TODO: Compute these variables.
    pred_gp_mean = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_gp_variance = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_gp = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))

    inv_val= get_inv_val(train_x,train_y,get_pce_kernel_fast)

    new_state= copy.deepcopy(init_state)
    for t in range(NUM_DATAPOINTS_PER_EPOCH):

        state= np.array(augmented_state(new_state,action_traj[t]))
        mean_set= np.zeros((train_y.shape[1],))
        variance_set = np.zeros((train_y.shape[1],))
        for k in range(train_y.shape[1]):

            K_star = get_pce_kernel_fast(train_x ,state.reshape((1,-1)),k)
            mean = K_star.T @ inv_val[k,:,:] @ train_y[:, k].reshape((-1,1))
            K_star_star = get_pce_kernel_fast(state.reshape((1, -1)), state.reshape((1, -1)), k)

            variance = K_star_star - K_star.T @ inv_val[k,:,:] @ K_star
            new_state[k] = new_state[k] + mean
            mean_set[k] = mean
            variance_set[k] = (variance)
        pred_gp_mean[t] = mean_set
        pred_gp_variance[t] = variance_set
        rollout_gp[t] = new_state

    pred_gp_mean_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_gp_variance_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_gp_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))

    for j in range(NUM_TRAJ_SAMPLES):

        new_state = copy.deepcopy(init_state)
        mean_traj = np.zeros ((NUM_DATAPOINTS_PER_EPOCH,train_y.shape[1]))
        variance_traj = np.zeros ((NUM_DATAPOINTS_PER_EPOCH,train_y.shape[1]))
        state_traj = np.zeros ((NUM_DATAPOINTS_PER_EPOCH,train_y.shape[1]))
        #print("new_trajectory", '\n',init_state)
        for t in range(NUM_DATAPOINTS_PER_EPOCH):

            state = np.array(augmented_state(new_state , action_traj[t]))
            mean_set = np.zeros((train_y.shape[1],))
            variance_set = np.zeros((train_y.shape[1],))
            #print('start_state', new_state, '\n')
            for k in range(train_y.shape[1]):

                K_star = get_pce_kernel_fast(train_x, state.reshape((1, -1)), k)
                mean = (K_star.T @ inv_val[k, :, :] @ train_y [:, k].reshape((-1, 1)))
                K_star_star = get_pce_kernel_fast(state.reshape((1, -1)), state.reshape((1, -1)), k)
                variance = K_star_star - K_star.T @ inv_val[k, :, :] @ K_star
                s = rng.normal(mean, np.sqrt(variance))
                new_state[k] = new_state[k] + s
                mean_set[k] = (mean)
                variance_set[k] = (variance)

            mean_traj[t]= mean_set
            variance_traj[t]= variance_set
            state_traj[t]= new_state

            #print('new_state', new_state, '\n')

        pred_gp_mean_trajs[j]= mean_traj
        pred_gp_variance_trajs[j]= variance_traj
        rollout_gp_trajs[j]= state_traj

    return pred_gp_mean, pred_gp_variance, rollout_gp, pred_gp_mean_trajs, pred_gp_variance_trajs, rollout_gp_trajs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from cartpole_sim import CartpoleSim
    from policy import SwingUpAndBalancePolicy, RandomPolicy
    from visualization import Visualizer
    import cv2

    vis = Visualizer(cartpole_length=1.5, x_lim=(0.0, DELTA_T * NUM_DATAPOINTS_PER_EPOCH))
    swingup_policy = SwingUpAndBalancePolicy('policy.npz')
    random_policy = RandomPolicy(seed=12831)
    sim = CartpoleSim(dt=DELTA_T)

    # Initial training data used to train GP for the first epoch
    init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)
    ts, state_traj, action_traj = sim_rollout(sim, random_policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)

    delta_state_traj = state_traj[1:] - state_traj[:-1]

    train_x, train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)

    for epoch in range(NUM_TRAINING_EPOCHS):
        vis.clear()

        # Use learned policy every 4th epoch
        if (epoch + 1) % 4 == 0:
            policy = swingup_policy
            init_state = np.array([0.01, 0.01, 0.05, 0.05]) * rng.randn(4)
        else:
            policy = random_policy
            init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)

        ts, state_traj, action_traj = sim_rollout(sim, policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)

        delta_state_traj = state_traj[1:] - state_traj[:-1]


        (pred_gp_mean,
         pred_gp_variance,
         rollout_gp,
         pred_gp_mean_trajs,
         pred_gp_variance_trajs,
         rollout_gp_trajs) = predict_gp(train_x, train_y, state_traj[0], action_traj)

        for i in range(len(state_traj) - 1):
            vis.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])
            vis.set_gt_delta_state_trajectory(ts[:i+1], delta_state_traj[:i+1])

            if i == 0:
                vis.set_gp_cartpole_state(state_traj[i][3], state_traj[i][2])
                vis.set_gp_cartpole_rollout_state([state_traj[i][3]] * NUM_TRAJ_SAMPLES,
                                                  [state_traj[i][2]] * NUM_TRAJ_SAMPLES)
            else:
                vis.set_gp_cartpole_state(rollout_gp[i-1][3], rollout_gp[i-1][2])
                vis.set_gp_cartpole_rollout_state(rollout_gp_trajs[:, i-1, 3], rollout_gp_trajs[:, i-1, 2])

            vis.set_gp_delta_state_trajectory(ts[:i+1], pred_gp_mean[:i+1], pred_gp_variance[:i+1])

            if policy == swingup_policy:
                policy_type = 'swing up'
            else:
                policy_type = 'random'

            vis.set_info_text('epoch: %d\npolicy: %s' % (epoch, policy_type))

            vis_img = vis.draw(redraw=(i==0))
            cv2.imshow('vis', vis_img)

            if epoch == 0 and i == 0:
                # First frame
                video_out = cv2.VideoWriter('cartpole.mp4',
                                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                            int(1.0 / DELTA_T),
                                            (vis_img.shape[1], vis_img.shape[0]))

            video_out.write(vis_img)
            cv2.waitKey(int(1000 * DELTA_T))

        # Augment training data
        new_train_x, new_train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)
        train_x = np.concatenate([train_x, new_train_x])
        train_y = np.concatenate([train_y, new_train_y])
