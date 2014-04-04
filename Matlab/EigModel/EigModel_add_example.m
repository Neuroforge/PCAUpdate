clear all;
close all;

x = randn( 2,1000);
y = randn(2,500 );


a = -10*pi/180;
c = cos(a);
s = sin(a);
A = [c -s; s c] * [1 0; 0 3];
x = A*x;
x(2,:) = x(2,:) - 5;


a = 30*pi/180;
c = cos(a);
s = sin(a);
A = [c -s; s c] * [2 0; 0 1];
y = A*y;
y(1,:) = y(1,:) + 10;

eX = EigModel_make( x );
eY = EigModel_make( y );
e = EigModel_add( eX,eY );

figure
hold on;
plot( x(1,:), x(2,:), 'k.' ,'MarkerSize', 1);
plot( y(1,:), y(2,:), 'k.', 'MarkerSize', 1);
h = EigModel2d_draw( eX,'r');
set(h,'LineWidth',2);
h = EigModel2d_draw( eY,'r');
set(h,'LineWidth',2);

h = EigModel2d_draw( e,'k');
set(h,'LineWidth',2);

axis equal;
axis off

