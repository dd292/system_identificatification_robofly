clc 
clear all
close all
T= 0.01;
load('../traj2.mat');
states1= states';
actions1= actions';
data= iddata(states1,actions1,T);

Order= [9,3,9];%[ny,nu,nx]
Parameters = [2.0e-4,1e-6,4e-9,0,0,0.012];
Ts = 0; 
m = idnlgrey('fly_dynamics',Order,Parameters)
m.Parameters(1).Fixed = false;
m.Parameters(2).Fixed = false;
m.Parameters(3).Fixed = false;
m.Parameters(4).Fixed = false;
m.Parameters(5).Fixed = false;
%load('../traj3.mat');
%states2= states';
%actions2= actions';
%max(actions2)
%min(actions2)
opt= nlgreyestOptions('Display','on','SearchMethod','gn');
compare(data,m)
Force_track1={};
Force_track2={};
Force_track3={};
torque_track={};
m= nlgreyest(data,m,opt)

