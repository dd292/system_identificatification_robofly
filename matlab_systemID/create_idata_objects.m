clear all
close all
clc
T= 0.01;
load('traj1.mat');
states= states';
actions= actions';
data1= iddata(states,actions,T);
load('traj2.mat');
states= states';
actions= actions';
data2= iddata(states,actions,T);
load('traj3.mat');
states= states';
actions= actions';
data3= iddata(states,actions,T);
