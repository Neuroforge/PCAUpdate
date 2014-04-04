close all;
clear all;

n = 3;
N = 2*n;
x = rand( n, N );

m = (n-1):(n+2);
for i = 1:length(m)
  e{i} = EigModel_make( x(:,1:m(i)) );
  h{i} = EigModel_Mahalanobis( x, e{i} );

  u = EigModel_project( x, e{i} );
  d{i} = EigModel_make(  u(:,1:m(i)) );
  g{i} = EigModel_Mahalanobis( u, d{i} );
  
  a = h{i};
  b = g{i};
  [a;b]
  
end

y = x( :, 1:m(i) );
my = mean( y, 2 );
yy = y - repmat( my, 1, size(y,2) );
C= (1/size(y,2)) * yy *yy';

diag(yy' * inv(C) * yy)'
