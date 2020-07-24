function u= map_hardware(u)
max_f_l=1.5;
max_torque = (max_f_l/2)*7.5

Min_hardware= [-16.8196, -16.0350, -12.9855];
Max_hardware= [8.4098, 16.0350, 12.9855];
u(1) = ((u(1)-Min_hardware(1)) /(Max_hardware(1) - Min_hardware(1)))*max_f_l;
u(2) = (u(2)/Max_hardware(2))*max_torque;
u(3) = (u(3)/Max_hardware(3))*max_torque;
end


