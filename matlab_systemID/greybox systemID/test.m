function a= test(b)
c = b+1;
d = evalin('base','c')

d= d+1;
assignin('base','c',d);
a=d;
end