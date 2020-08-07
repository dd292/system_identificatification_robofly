clc; clear all; close all;

addpath('./Resources')

%% ****************************** Dynamics ********************************

n = 2; % Number of states
m = 1; % Number of control inputs
f_u = @dyn_motor_scaled; % Dynamics


% Discretize
deltaT = 0.01;
%Runge-Kutta 4
k1 = @(t,x,u) (  f_u(t,x,u) );
k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*deltaT/2,u) );
k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT,u) );
f_ud = @(t,x,u) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );

%% Collect data
rng(115123)
disp('Starting data collection')
Ntraj = 200; Nsim = 1000;

Cy = [0 1]; % Output matrix: y = Cy*x
nD = 1; % Number of delays
ny = size(Cy,1); % Number of outputs

% Random control input forcing
Ubig = 2*rand(Nsim, Ntraj) - 1;

% Random initial condition
Xcurrent = (rand(n,Ntraj)*2 - 1);
X = []; Y = []; U = [];
zeta_current = [Cy*Xcurrent ; NaN(nD*(ny+m),Ntraj)];

% Delay-embedded "state" 
% zeta_k = [y_{k} ; u_{k-1} ; y_{k-1} ... u_{k-nd} ; y_{k-nd} ];

n_zeta = (nD+1)*ny + nD*m; % dimension of the delay-embedded "state"
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent,Ubig(i,:));
    zeta_prev = zeta_current;
    zeta_current = [[Cy*Xnext ; Ubig(i,:)] ; zeta_current( 1:end-ny-m , : ) ];
    if(i > nD)
        X = [X zeta_prev];
        Y = [Y zeta_current];
        U = [U Ubig(i,:)];
    end
    Xcurrent = Xnext;
end

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

Tmax = 1;
Nsim = Tmax/deltaT;

uprbs = (2*myprbs(Nsim,0.5) - 1);
u_dt = @(i)(  uprbs(i+1) );
f_cont_d = @(t,xx)( f_ud(t,xx,u_dt(t)) );

x0 = rand(2,1)-0.5;
x = x0;

% Delayed initial condition (assume random control input in the past)
xstart = [Cy*x ; NaN(nD*(ny+m),1)];
for i = 1:nD
    urand = 2*rand(m,1) - 1;
    xp = f_ud(0,x,urand);
    xstart = [Cy*xp ; urand; xstart(1:end-ny-m)];
    x = xp;
end

% Inital conditions
x_true = xp;
xlift = liftFun(xstart);

% Simulation
for i = 0:Nsim-1
    
    % True dynamics
    x_true = [x_true, f_ud(0,x_true(:,end),u_dt(i)) ];
    
    % Koopman predictor
    xlift = [xlift Alift*xlift(:,end) + Blift*u_dt(i)];
end

figure
stairs([0:Nsim-1]*deltaT,u_dt(0:Nsim-1),'linewidth',2); hold on
title('Control input'); xlabel('time [s]')

x_ref = Cy*x_true;
x_koop = Clift(1,:)*xlift;

figure
lw_koop = 3;
plot([0:Nsim]*deltaT,x_ref,'-b','linewidth', lw_koop); hold on
plot([0:Nsim]*deltaT,x_koop, '--r','linewidth',lw_koop)
LEG = legend('True','Koopman');
set(LEG,'Interpreter','latex','location','northeast','fontsize',30)
set(gca,'FontSize',25);
axis([0 1 -1.3 0.5])


