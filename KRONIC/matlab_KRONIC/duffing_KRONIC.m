clc; clear all; close all;

% Parameters
dt = 0.001;
tspan = dt:dt:10;
x0 = [0; -2.8];

f = @(t,x)([x(2); x(1) - x(1)^3]);
H = @(x)( (1/2)*x(2).^2-(1/2)*x(1).^2 + (1/4)*x(1).^4 );
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

usesine = 0;
polyorder = 4;
nvar = 2;

%% Trajectory data

[t,y] = ode45(f,tspan,x0,ode_options);

% using fourth order central difference
dy = zeros(length(y)-5,nvar);
for i=3:length(y)-3
    for k=1:nvar
        dy(i-2,k) = (1/(12*dt))*(-y(i+2,k)+8*y(i+1,k)-8*y(i-1,k)+y(i-2,k));
    end
end
y = y(3:end-3,1:nvar);
t = t(3:end-3);

Hy = zeros(length(y),1);
for k=1:length(y)
    Hy(k) = H(y(k,:));
end

figure; hold on, box on
plot(t,y(:,1),'-','Color',[0,0,0.7],'LineWidth',2)
plot(t,y(:,2),'-','Color',[0,0.7,0],'LineWidth',2)
legend('x1','x2')
xlabel('t'), ylabel('xi')
set(gca,'xtick',[0:2:10])
set(gca,'FontSize',16)
set(gcf,'Position',[100 100 225 200])
set(gcf,'PaperPositionMode','auto')


% Construct libraries
Theta = buildTheta(y,nvar,polyorder,usesine);
Gamma = buildGamma(y,dy,nvar,polyorder,usesine);

% Compute SVD
[U,S,V] = svd(0*Theta - Gamma,'econ');

% Least-squares Koopman
K = pinv(Theta)*Gamma;
K(abs(K)<1e-12) = 0;
[T,D] = eig(K);
D = diag(D);
[~,IX] = sort(abs(D),'ascend');

% Compute eigenfunction
xi0 = V(:,end);             % from SVD
xi0(abs(xi0)<1e-12) = 0;

D(IX(1))
xi1 = T(:,IX(1));%+Tls(:,IX(2));  % from least-squares fit
xi1(abs(xi1)<1e-12) = 0; 

% Plot evolution of eigenfunction = Hamiltonian
if length(Hy)~=length(t)
    t = 1:length(Hy);
end

normalized1 = (Theta)*(xi0)./norm((Theta)*(xi0));
normalized2 = -(Theta*xi1)./norm(Theta*xi1);


% figure()
% plot(t,normalized2,'-b', 'LineWidth',8,'Color',[0,0,1])
% title('Evolution')
% ylim([0.01-1e-4 0.01+2e-4]), xlim([min(t) max(t)])
% set(gca,'xtick',[0:2:10])
% set(gca,'FontSize',16)
% set(gcf,'Position',[100 100 260 200])
% set(gcf,'PaperPositionMode','auto')

figure; hold on, box on
ph(1) = plot(t,Hy./norm(Hy),'-k', 'LineWidth',18,'Color',[0.7,0.7,0.7]);
ph(2) = plot(t,(Theta)*(xi0)./norm((Theta)*(xi0)),'-b', 'LineWidth',8,'Color',[0,0,1]);
ph(3) = plot(t,-(Theta)*(xi1)./norm((Theta)*(xi1)),'--', 'LineWidth',8,'Color',[0,0.5,0]);
xlabel('t'), ylabel('E')
ylim([0.01-1e-4 0.01+2e-4]), xlim([min(t) max(t)])
set(gca,'xtick',[0:2:10])
legend(ph,'True place', 'SVD place', 'LS place')
set(gca,'FontSize',16)
set(gcf,'Position',[100 100 260 200])
set(gcf,'PaperPositionMode','auto')


% % Plot error 
dstep = 5;
clear ph
figure; hold on, box on
tmp = Gamma*xi0;
ph(1) = plot(t(1:dstep:end),tmp(1:dstep:end),'-k', 'LineWidth',2);%,'Color',[0,0,1]);
tmp = -Gamma*xi1;
ph(2) = plot(t(1:dstep:end),tmp(1:dstep:end),'-r', 'LineWidth',2);%,'Color',[0,0.5,0]);
xlabel('t'), ylabel('Gamma xi')
xlim([min(t) max(t)])
%ylim([min(Gamma*xi1)+0.2*min(Gamma*xi1) max(Gamma*xi1)+0.5*max(Gamma*xi1)])
legend(ph,'SVD p', 'LS  p')
set(gca,'xtick',[0:2:10])
set(gca,'FontSize',16)
set(gcf,'Position',[100 100 225 200])
set(gcf,'PaperPositionMode','auto')
% 
% err0 = (Hy./norm(Hy)-(Theta)*(xi0)./norm((Theta)*(xi0)))./(Hy./norm(Hy));
% err1 = (Hy./norm(Hy)-(Theta)*(xi1)./norm((Theta)*(xi1)))./(Hy./norm(Hy));
% figure, plot(err0,'-r'),hold on, plot(err1,'--b')
% axis tight

