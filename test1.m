Nx = 12;
Ny = 3;

dx = 100e3;
dy = dx;
dmin = min([dx,dy]);

x = 0:dx:dx*(Nx-1);
y = 0:dy:dy*(Ny-1);
y0 = 3000000;
[Y,X] = meshgrid(y,x);

omega = 7.2921e-5;
latitude = 35;
g = 9.81;
R = 6371e3;

f = @(phi) 2*omega*sin(phi);

coriolis = f(latitude*pi/180) + (2*omega/R)*(Y-y0);

W = 10000 - 500*tanh(3e-06*(Y-y0));

[dWdy, dWdx] = gradient(W, dy, dx);
u_geo = -g./coriolis.*dWdy;
v_geo = g./coriolis.*dWdx;

hu = W.*u_geo;
hv = W.*v_geo;