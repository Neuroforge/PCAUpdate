% Example of how to use (some of) the library functions

% clean the slate
clear all;
close all;

% first demonstrate how to make an  eigenmodel
% for a set of randomly distributed data.

n = 2;        % dimension of each point
N = 1000; % number of points

% make the data
randn('state', sum(100*clock) ); % set a random seed
x = randn(n,N); % each element in x is a random number in [0,1]
                          % so each datum is inside a square
shift = rand(n,1); % a shift for the data
angle = rand *  2 *pi; 
rotate = [cos(angle) -sin(angle);...
               sin(angle) cos(angle)];
scale = [4/rand 0;...
               0 1/rand];
% transform the data
x = (rotate*scale)*(x - repmat(shift,1,N));

% make an Eigenmodel (perform EVD on data)
e = EigModel_make( x );

% visualise results
figure;
hold on;
axis equal;
plot( x(1,:), x(2,:), 'k.');
h = EigModel2D_draw( e, 'r' ); % this works only for EigModels with 2 eigenvectors
set(h,'LineWidth',2);

% just replace the data x by vertices from a polygon!
% :-)

return;

