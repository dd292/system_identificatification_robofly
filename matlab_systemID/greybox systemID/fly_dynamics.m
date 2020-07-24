
function [qdot,y] = fly_dynamics(t, x,u,drag,mass,Moment, torque_noise, force_noise,r_w,random)


% robofly params
p.winglength = 0.0132;% metres 
p.l = .045; % length of fly
p.h = .003; % thickness of body
p.J = Moment;%5e-9; % 1d version
p.Jmat = diag([p.J, p.J p.J]); %from the robofly expanded controllere-9
p.b_w = drag; % aero drag on wings from wind tunnel tests, Ns/m (to learn)
p.c_w = (p.h/2 + p.winglength * 2./3)^2 * p.b_w; % rot drag coeff around z [Nsm]
p.r_w = r_w;%.012; % z-dist from ctr of wings to ctr of mass 
p.m = mass;% 130e-6; %mass of triplef2ly
p.g = 9.81; 
p.max_f_l = 1.5 * p.m * p.g; 
p.max_torque = (p.max_f_l/2)*p.l/2;  
p.ks = 0;%0.5e-6; % tether stiffness
p.force_bias_x = 0; %0.0001; %N
p.torque_bias_y = 0; %-0.1e-6; %Nm
p.gyro_bias_y = 0; %0.1; % rad/s
% leave the following three zero for this simulation:
p.force_bias_y = 0; %N
p.torque_bias_x = 0; %Nm
p.gyro_bias_x = 0; % rad/s


%hardware input mapping
u = map_hardware(u,p);
% given state and inputs u from controller, calculate derivatives for ode45
% integrator. 
vbody = x(1:3); 
theta = x(4:6);
omegabody = x(7:9);
%posworld = q(7:9); % not used in calculation
f_l = u(1);
tau_c = [u(2);u(3);0]; 

R = rot_matrix(theta);
W = omega2thetadot_matrix(theta);

%if nargin < 6, 
f_disturb = zeros(3,1);
%if nargin < 5,
tau_disturb = zeros(3,1);

% aerodynamic forces and torques
[f_d, tau_d] = fly_aerodynamics(vbody, omegabody, p);

% forces (f) (body coords)
f_g = R' * [0, 0, - p.g * p.m]'; % gravity
f_l = [0, 0, f_l]'; % control force
%f_disturb = [p.force_bias_x, p.force_bias_y, 0]'; 

f = f_l + f_g + f_disturb + f_d +force_noise; 

% moments/torues (tau) (body coords)
%tau_disturb = [p.torque_bias_x, p.torque_bias_y, 0]'; 
tau = tau_c + tau_d + tau_disturb - p.ks * [theta(1), theta(2), 0]'+torque_noise;

fictitious_f = p.m * cross(omegabody, vbody); %[0; omega; 0], [q(4); 0; q(6)]); % omega x v
fictitious_tau = cross(omegabody, p.Jmat * omegabody);
% 
% Force_track1 = evalin('base','Force_track1');
% Force_track1 =[Force_track1, fictitious_tau(1)];
% assignin('base','Force_track1',Force_track1);
% 
% Force_track2 = evalin('base','Force_track2');
% Force_track2 =[Force_track2, fictitious_tau(2)];
% assignin('base','Force_track2',Force_track2);
% 
% 
% Force_track3 = evalin('base','Force_track3');
% Force_track3 =[Force_track3, fictitious_tau(3)];
% assignin('base','Force_track3',Force_track3);

% geometric
%xdotworld = R * vbody; 
vdotbody = 1/p.m * (f - fictitious_f); 
thetadot = W * omegabody; 
omegadotbody = p.Jmat \ (tau - fictitious_tau);
qdot = [vdotbody; thetadot; omegadotbody];
y=qdot;
end

function u= map_hardware(u,p)

Min_hardware= [-16.8196, -16.0350, -12.9855];% took from the data
Max_hardware= [8.4098, 16.0350, 12.9855];% took from the data
u(1) = ((u(1)-Min_hardware(1)) /(Max_hardware(1) - Min_hardware(1)))*p.max_f_l;
u(2) = (u(2)/Max_hardware(2))*p.max_torque;
u(3) = (u(3)/Max_hardware(3))*p.max_torque;
end


function [f_l, tau_c] = wing_force_step(t, q, ~, p)
% simulation to examine if phase offset can produce yaw rotation
flapping_f = 120; % rad/s
flapping_w = flapping_f * 2 * pi; 
right_wing_phase_offset = 60 * pi / 180; % rad
lcpwing = .01; % length to center of pressure of wing; assume 2/3 lwing. 
fl_amplitude_wing = p.m * p.g / 2; % assume lift goes as a sinusoid at 2f
fd_amplitude = fl_amplitude_wing; % assume drag = lift
if 0
    left_wing_drag_force = fd_amplitude * sin(flapping_w * t + 0);
    right_wing_drag_force = fd_amplitude * sin(flapping_w * t + right_wing_phase_offset);

    left_wing_yaw_torque =  lcpwing * left_wing_drag_force; 
    right_wing_yaw_torque = -lcpwing * right_wing_drag_force;
    yaw_torque = left_wing_yaw_torque + right_wing_yaw_torque;

    left_wing_pitch_torque =  -p.r_w * left_wing_drag_force; 
    right_wing_pitch_torque = -p.r_w * right_wing_drag_force;
    pitch_torque = left_wing_pitch_torque + right_wing_pitch_torque;


    left_wing_lift_force = fl_amplitude_wing * ...
        (1 +  sin(2*flapping_w * t + 0));
    right_wing_lift_force = fl_amplitude_wing * ...
        (1 + sin(2*flapping_w * t + right_wing_phase_offset));
    f_l = left_wing_lift_force + right_wing_lift_force; 

    left_wing_roll_torque = left_wing_lift_force * lcpwing; 
    right_wing_roll_torque = right_wing_lift_force * (-lcpwing); 
    roll_torque = left_wing_roll_torque + right_wing_roll_torque;
else
    f_l = p.m * p.g; 
    pitch_torque = fd_amplitude * lcpwing * sin(flapping_w * t); 
    %roll_torque = fd_amplitude * lcpwing * cos(flapping_w * t); 
    roll_torque = fd_amplitude * lcpwing * sin(2*flapping_w * t + -right_wing_phase_offset); 
    yaw_torque = fd_amplitude * lcpwing * sin(2*flapping_w * t + right_wing_phase_offset); 
end

% add stabilizing term
omegabody = q(4:6); 
tau_c = -1e-7 * omegabody; 
tau_c = clip(tau_c, p.max_torque); 
tau_c = tau_c + [roll_torque; pitch_torque; yaw_torque];
% tau_c = tau_c + [roll_torque; 0; yaw_torque];
% tau_c = [roll_torque; 0; yaw_torque];

end

function clipped = clip(x, absmax)
    clipped = max(min(x, absmax), -absmax);
end

function A = estimate_state_jacobian(dynamics, t, q, varargin)
n_states = length(q); 
A = zeros(n_states); 
dq = sqrt(eps) * q; % optimal dq from wikipedia on numerical diff. 
for idx = 1:n_states
    dqvec = zeros(n_states, 1); 
    if q(idx) == 0
        dqvec(idx) = 1e-15; 
    else
        dqvec(idx) = dq(idx); 
    end
    A(:, idx) = (dynamics(t, q+dqvec, varargin{:}) - dynamics(t, q-dqvec, varargin{:}))/...
        (2 * dqvec(idx));
end
end

function qdot = fly_dynamics_wrapper(t, q, u, p)
% break out parts
torque_bias_estimate = q(13:15); 
qdot = zeros(18, 1); % assume no bias changes, update the rest

qdot(1:12) = fly_dynamics(t, q(1:12), u, p, torque_bias_estimate); 
% qdot(13:18) = 0; % assume no bias changes
end

function y = measurement_model(~, q, p)
gyro_bias = q(16:18); 
y = [q(4:6) + gyro_bias; q(10:12)];
end


function c = cross(a,b) % fast cross because default cross func is slow
c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)];
end


function R = rot_matrix(eulerAngles)
% R matrix to convert 3-vectors in body coords to world coords
% v = Rv' where v is in world frame and v' is in body frame


% XYZ/zyx (airplane) convention
% theta1 = thetaz = thetaX; theta2 = thetay = thetaY; theta3 = thetax =
% thetaZ. got it? : )
% no that is dumb. the order is different, but make theta1 always the
% rotation around x. 
cz = cos(eulerAngles(3)); 
cy = cos(eulerAngles(2));
cx = cos(eulerAngles(1));
sz = sin(eulerAngles(3));
sy = sin(eulerAngles(2));
sx = sin(eulerAngles(1));
R = [cz*cy,   cz*sy*sx - cx*sz,   sz*sx + cz*cx*sy; 
     cy*sz,   cz*cx + sz*sy*sx,   cx*sz*sy - cz*sx;
     -sy,    cy*sx,              cy*cx];  

% ZYX (vicon) convention
% convention here is ZYX convention: the three coordinates in euler_theta  
% mean rotate around world Z, then world Y, then world X 
% alternately, rot around body x, then new body y, then new body z
% (the latter two are equivalent)
% theta1 = thetax = thetaZ; theta2 = thetay = thetaY; theta3 = thetaz =
% thetaX    
% c1 = cos(eulerAngles(1)); 
% c2 = cos(eulerAngles(2));
% c3 = cos(eulerAngles(3));
% s1 = sin(eulerAngles(1));
% s2 = sin(eulerAngles(2));
% s3 = sin(eulerAngles(3));
% R = [c2.*c3, -c2.*s3, s2;
%      c1.*s3 + c3.*s1.*s2, c1.*c3 - s1.*s2.*s3, -c2.*s1;
%      s1.*s3 - c1.*c3.*s2, c3.*s1 + c1.*s2.*s3, c1.*c2];
end

function rotatedVectors = rotVectors(eulerAngles, vectors)
% rotate vectors. 
% angles is 3xn or 3x1, vectors is 3xn 
% rotates a vector v' (vectors) given in body-frame coordinates to 
% v (rotatedVectors) given in world-frame coords according to 
% v = Rv'

dim = size(eulerAngles);
n = dim(2);
dim = size(vectors); 
nv = dim(2);
if n == 1
    eulerAngles = repmat(eulerAngles, [1, nv]); 
end
cz = cos(eulerAngles(3,:)); 
cy = cos(eulerAngles(2,:));
cx = cos(eulerAngles(1,:));
sz = sin(eulerAngles(3,:));
sy = sin(eulerAngles(2,:));
sx = sin(eulerAngles(1,:));
R1 = [cz.*cy;   cz.*sy.*sx - cx.*sz;   sz.*sx + cz.*cx.*sy];
R2 = [cy.*sz;   cz.*cx + sz.*sy.*sx;   cx.*sz.*sy - cz.*sx];
R3 = [-sy;    cy.*sx;              cy.*cx];  


% ZYX (vicon) convention
% rows of R
% R1 = [c2.*c3, -c2.*s3, s2];
% R2 = [c1.*s3 + c3.*s1.*s2, c1.*c3 - s1.*s2.*s3, -c2.*s1];
% R3 = [s1.*s3 - c1.*c3.*s2, c3.*s1 + c1.*s2.*s3, c1.*c2];

% do a dot product by summing vertically (along 1st dimension)
rotatedVectors = [
    sum(R1.*vectors,1); ...
    sum(R2.*vectors,1); ...
    sum(R3.*vectors,1)]; 
end


function transformedVectors = transformVectors(eulerAngles, translation, vectors)
% rotate and translate vectors. 
% eulerAngles is 3xn or 3x1, translation is 3xn or 3x1, vectors is 3xn 
% rotates a vector v' (vectors) given in body-frame coordinates 
% a vector v to world-frame coords  
% and translates it by t (translation)
% v = Rv' + t

dim = size(translation);
n = dim(2);
if n == 1
    dim = size(vectors); 
    nv = dim(2);
    translation = repmat(translation, [1, nv]); 
end
transformedVectors = rotVectors(eulerAngles, vectors) + translation; 
end

function W = omega2thetadot_matrix(euler_theta)
% transform euler angle rates to angular rot rate vector omega
% thetadot = W * omega
% this needs tobe corrected for 3d case for the current euler angle convention. 
st1 = sin(euler_theta(1));
ct1 = cos(euler_theta(1));
tt2 = tan(euler_theta(2));
ct2 = cos(euler_theta(2));

% XYZ (airplane) convention
W = [1, st1*tt2, ct1*tt2; 
    0, ct1, -st1;
    0, st1/ct2, ct1/ct2];

% ZYX (vicon) convention
% needs fixing
% W = [c3/c2,   -s3/c2,     0; 
%      s3,      c3,         0; 
%      -c3*t2,  s3*t2,      1];  

end


function [f_d, tau_d] = fly_aerodynamics(v, omega, p)
% calculate stroke-averaged forces due to aerodynamic drag on flapping 
% wings. assumes force is applied at point r_w away from center of mass in
% z-direction in body-coordinates. v_w is the vel at that point. 
% assumes drag force f_d = -b_w * v_w. this has been tested in a wind
% tunnel for x- and y-directions but not z-direction
r_w = [0, 0, p.r_w]';
v_w = v + cross(omega, r_w); % vel at midpoint of wings
f_d = - p.b_w * v_w;  
tau_d = cross(r_w, f_d); %p.r_w * f_d(1); % r_w cross f_d
end

function render_body(eulerAngles, position, and_wings)

if nargin < 3
    and_wings = 1;
end
if nargin < 2
    position = zeros(3,1); 
end
if nargin < 1
    eulerAngles = zeros(3,1); 
end

persistent moving_geometry fixed_geometry 
if isempty(moving_geometry)
    % initialize persistent vars
    b.l = 3e-3; 
    b.w = 2e-3; 
    b.h = 13e-3; 
    moving_geometry.body.patch = patch('EraseMode','normal');
    moving_geometry.body.verts = [
        b.l/2 * [-1, -1, 1, 1, -1, -1, 1, 1]; % length
        b.w/2 * [-1, 1, 1, -1, -1, 1, 1, -1]; % width
        b.h/2 * [-1, -1, -1, -1, 1, 1, 1, 1]]; % height
    moving_geometry.body.faces = [1 2 3 4; 2 6 7 3;  4 3 7 8; ...
                                  1 5 8 4; 1 2 6 5; 5 6 7 8];
    moving_geometry.shadow.patch = patch('EraseMode','normal');
    moving_geometry.shadow.verts = [
        -b.l/2, -b.w/2, .001; 
        b.l/2, -b.w/2, .001; 
        b.l/2, b.w/2, .001; 
        -b.l/2, b.w/2, .001]';
    moving_geometry.shadow.faces = 4:-1:1; 
    if and_wings
        vertr = 1e-3*[14.9           25         27.5         28.4           29         29.2         29.3         29.3         29.2         28.8         27.7         25.7         22.6         21.1         20.3         19.3         18.4         17.5         14.9;
                           35.2         35.2         35.2         35.1         34.9         34.7         34.6         34.5         34.3         33.8         33.1           32         30.7         30.2         30.1         30.1         30.4         31.1         33.6];
        % put into x-y-z coordinates
        vertr = [zeros(1, length(vertr)); vertr];
        % recenter and position relative to fly correctly
        vertr = vertr ...
            - repmat(vertr(:,1), [1, length(vertr)]) ...
            + repmat([0; b.w/2 + 0.05e-3; b.h/2], [1, length(vertr)]); 
        vertl = [vertr(1,:); -vertr(2,:); vertr(3,:)]; 
        moving_geometry.rightwing.patch = patch('EraseMode','normal');
        moving_geometry.rightwing.verts = vertr; 
        moving_geometry.rightwing.faces = length(vertr):-1:1;
        
        moving_geometry.leftwing.patch = patch('EraseMode','normal');
        moving_geometry.leftwing.verts = vertl;
        moving_geometry.leftwing.faces = length(vertl):-1:1;
        
    end
    fixed_geometry.floor.patch = patch('EraseMode','normal');
    fs = 50e-3; % floor size m
    fixed_geometry.floor.verts = [-fs, -fs, 0; fs, -fs, 0; fs, fs, 0; -fs, fs, 0]'; 
    fixed_geometry.floor.faces = 4:-1:1; 
end
    
fieldnames_ = fieldnames(moving_geometry); 
for idx = 1:length(fieldnames_)
    fieldname = fieldnames_{idx};
    verts = moving_geometry.(fieldname).verts; 
    if strcmp(fieldname, 'shadow')
        % special case for shadow, only rotate aroud z and don't shift z
        shadow_eulerAngles = [0; 0; eulerAngles(3)];
        shadow_position = [position(1:2); 0]; 
        transformed_verts = transformVectors(shadow_eulerAngles, shadow_position, verts);
        color = 'black'; 
    else
        transformed_verts = transformVectors(eulerAngles, position, verts); 
        color = 'white'; 
    end
    set(moving_geometry.(fieldname).patch, ...
        'Faces', moving_geometry.(fieldname).faces, ...
        'Vertices',transformed_verts', ...
        'FaceColor',color)
end

fieldnames_ = fieldnames(fixed_geometry); 
for idx = 1:length(fieldnames_)
    fieldname = fieldnames_{idx};
    set(fixed_geometry.(fieldname).patch, ...
        'Faces', fixed_geometry.(fieldname).faces, ...
        'Vertices',fixed_geometry.(fieldname).verts', ...
        'FaceColor','white')
end
     
%axis tight
axis([-.06 .06 -.06 .06 0 .09])
end
