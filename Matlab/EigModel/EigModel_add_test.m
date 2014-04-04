% EigModel_make_add.m

clear all;
close all;

N = 1;
M = 1;
n = 1;
X = gsamp( 1, 1, N )';
Y = gsamp( 0, 0.5, M )';

% make single models...
mX = EigModel_make( X ); % ...from one data set...
mY = EigModel_make( Y ); % ...from another...
mZ = EigModel_make( [X,Y] ); % and their concatenation (to get a control).

% add two individual models
mXY = Eigmodel_add( mX, mY );

% compare the results
mXY.N - mZ.N
mXY.org - mZ.org
mXY.val - mZ.val
mXY.vct - mZ.vct

return;

% Test for simple addition
n = 2; % number of dimensions

N = 200;
X = gsamp( [0,0], [1, 0; 0 1], N )'; % from NetLab
S = [4 0; 0 2];
a = 30*pi/180; R = [cos(a) -sin(a); sin(a) cos(a)];
T = [5; 6];
X = R*S*X + repmat(T,1,N);

N = 300;
Y = gsamp( [0,0], [1, 0; 0 1], N )'; % from NetLab
S = [3 0; 0 1];
a = -20*pi/180; R = [cos(a) -sin(a); sin(a) cos(a)];
T = [-2; 6];
Y = R*S*Y + repmat(T,1,N);

mX = EigModel_make( X );
mY = EigModel_make( Y );
mZ = EigModel_make( [X,Y] );
mXY = Eigmodel_add( mX, mY );


figure;
hold on;
axis equal;

plot( X(1,:), X(2,:), 'color', [0.75 0.75 1], 'Linestyle', 'none' , 'Marker', '.');
plot( Y(1,:), Y(2,:), 'color', [1 0.75 0.75], 'Linestyle', 'none' , 'Marker', '.');

hX = EigModel2D_draw( mX, 'b' );
set(hX, 'LineWidth', 2 );
hY = EigModel2D_draw( mY, 'r' );
set(hY ,'LineWidth', 2 );


hZ = EigModel2D_draw( mZ, 'k' );
set(hZ,'LineWidth',3);

hXY = EigModel2D_draw( mXY, 'g' );
set(hXY,'LineWidth',1);

legend( [hX(1) hY(1) hZ(1) hXY(1)], '\Omega[X]', '\Omega[Y]', ...
    '\Omega[X \cup Y]', '\Omega[X] \oplus \Omega[Y]');