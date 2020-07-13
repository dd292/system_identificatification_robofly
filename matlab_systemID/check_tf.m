clc 
points= size(actions);
current_tf= tf1;
input= actions';
time= 0:0.01:points(1)*0.01;
time = time(1:end-1);
[y,t]= lsim(current_tf, input, time);