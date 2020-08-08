clc; clear all; close all;

addpath('./resources')

%% ****************************** Dynamics ********************************

nstates = 2; % Number of states
nctrl = 1; % Number of control inputs
f_u = @dyn_motor_scaled; % Dynamics

% Discretize
dt = 0.01;
%Runge-Kutta 4
k1 = @(t,x,u) (  f_u(t,x,u) );
k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*dt/2,u) );
k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*dt/2,u) );
k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*dt,u) );
rk4_step = @(t,x,u) ( x + (dt/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );

%% Collect data
disp('Starting data collection')
% Ntraj = 200; Nsim = 1000;

Tmax = 1000;
Nsim = Tmax/dt;
n = [0:Nsim-1]*dt;

% Random control input forcing
u = 2*rand(Nsim,1) - 1;

% Random initial condition
x_current = (rand(nstates,1)*2 - 1);
X = []; Y = []; U = [];

train_traj = NaN(nstates,Nsim);

for i = 1:Nsim
    train_traj(:,i) = x_current;
    x_current = rk4_step(0,x_current,u(i));
end

Cy = [0 1]; % Output matrix: y = Cy*x
nD = 1; % Number of delays
ny = size(Cy,1); % Number of outputs

n_zeta = (nD+1)*ny + nD*nctrl; % dimension of the delay-embedded "state"

% Build Hankel Matrix
H = NaN(n_zeta,Nsim-nD);
H(1,:) = Cy*train_traj(:,1:end-nD); 
H(2,:) = Cy*train_traj(:,2:end);
H(3,:) = u(1:end-1);

% Build delay embedded X and Y Matrices
X = H(:,1:end-1); Y = H(:,2:end);
% Build U Matrix 
U = u(1:end-nD-ny); U = U';

fprintf('Data collection DONE \n');

%% Basis functions
basisFunction = 'rbf';
Nrbf = 100;
cent = rand(n_zeta,Nrbf)*2 - 1; % RBF centers
rbf_type = 'thinplate';
theta_max = pi;
liftFun = @(xx)( [xx;rbf(xx,cent,rbf_type)] );
Nlift = Nrbf + n_zeta;
%% Lift
disp('Starting LIFTING')

Xlift = liftFun(X);
Ylift = liftFun(Y);

%% Regression

disp('Starting REGRESSION for A,B,C')

W = [Ylift ; X];
V = [Xlift ; U];
VVt = V*V';
WVt = W*V';
ABC = WVt * pinv(VVt);
Alift = ABC(1:Nlift,1:Nlift);
Blift = ABC(1:Nlift,Nlift+1:end);
Clift = ABC(Nlift+1:end,1:Nlift);
fprintf('Regression for A, B, C DONE \n');

% Residual
fprintf( 'Regression residual %f \n', norm(Ylift - Alift*Xlift - Blift*U,'fro') / norm(Ylift,'fro') );

Nsim_test = 2000;

% random forcing
u_test = 2*rand(Nsim_test,1) - 1;
% random sample
u_test_sample = u_test(nctrl,1);
% random initial condition
x0_test = rand([nstates,1])*2 - 1;

delay_init = [x0_test;u_test_sample];
xlift = liftFun(delay_init);

x_true = x0_test;

% Simulation
for i = 1:Nsim_test-1 
    
    % True dynamics
    x_true = [x_true, rk4_step(0,x_true(:,end),u_test(i)) ];
    
    % Koopman predictor
    xlift = [xlift Alift*xlift(:,end) + Blift*u_test(i)];
end

x_koop = Clift*xlift; % Koopman predictions
x_true = Cy*x_true;

figure
lw_koop = 3;
plot([0:Nsim_test-1]*dt,x_true,'-b','linewidth', lw_koop); hold on
plot([0:Nsim_test-1]*dt,x_koop(1,:), '--r','linewidth',lw_koop)
LEG = legend('True','Koopman');
set(LEG,'Interpreter','latex','location','northeast','fontsize',30)
set(gca,'FontSize',25);
axis([0 1 -1.3 0.5])


