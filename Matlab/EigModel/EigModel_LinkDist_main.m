close all;
clear all;

% dimension of space
n = 2;

% make data
X = randn( n,  round(100+900*rand));
if 0
  % affine transform on points
  a = rand*2*pi;
  s = rand(n,1);
  u = 3*(rand(n,1)-0.5);
else
  % default point locations
  a = 0;
  s = ones(n,1);
  u = zeros(n,1);
end
X = [cos(a) -sin(a); sin(a) cos(a)]*diag(s)*X + u*ones(1,size(X,2));

% make Eigenmodel from data
eX = EigModel_make( X );
hX = sqrt(EigModel_Mahalanobis( X, eX ));

% use the SAME data to test the linakge distance from a patch to itself
Y = X;
eY = EigModel_make( Y );

% the linkage distance
d0 = (eY.N*sum(sqrt( EigModel_Mahalanobis( Y, eX ))) +eX.N*sum(sqrt( EigModel_Mahalanobis( X, eY )))) / (eX.N+eY.N);

% repeat this
for i = 1:100
  X = randn( n,  round(100+900*rand));
  Y = X;
  eX = EigModel_make( X );
  eY = EigModel_make( Y );
  devX(:,i) = eX.val;
  devY(:,i) = eY.val;
  d(i) = (eY.N*sum(sqrt( EigModel_Mahalanobis( Y, eX ))) +eX.N*sum(sqrt( EigModel_Mahalanobis( X, eY )))) / (eX.N+eY.N);
end

[ mean( devX, 2 ) std( devX, [], 2 ) ]
[ mean(d) std(d) ]

if 0
  % make a picture of this model
  figure;
  axis equal;
  hold on;
  mask1 = hX <= 1;
  mask2 = (hX <= 2) & ~mask1;
  mask3 = (hX <= 3) & ~(mask1 | mask2);
  mask4 = ~(mask1 | mask2|mask3);
  plot( X(1,mask1), X(2,mask1), 'r.' );
  plot( X(1,mask2), X(2,mask2), 'g.' );
  plot( X(1,mask3), X(2,mask3), 'b.' );
  plot( X(1,mask4), X(2,mask4), 'k.' );
  EigModel2D_draw( eX, 'r' );
  EigModel2D_draw( eX, 'g' , 2 );
  EigModel2D_draw( eX, 'b' , 3 );
end

return;



a = 0;
s = ones(n,1);
u = zeros(n,1);

% Y = randn( n, round(100+900*rand) );
% % a = rand*2*pi;
% % s = rand(n,1);
% % u = 3*(rand(n,1)-0.5);
% a = 0;
% s = ones(n,1);
% u = zeros(n,1);
% Y = [cos(a) -sin(a); sin(a) cos(a)]*diag(s)*Y + u*ones(1,size(Y,2));

Y = X;



% Compute the linkage distance



% figure;
% axis equal;
% hold on
% plot( X(1,:), X(2,:) , 'k.' );
% plot( Y(1,:), Y(2,:) , 'b.' );
