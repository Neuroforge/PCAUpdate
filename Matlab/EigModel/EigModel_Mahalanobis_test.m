clear all;
close all;

n = 4; % number of dimensions
N = 2000; % number of points


x = gsamp(  zeros(1,n), eye(n), N )' ;
S = diag(4*rand(1,n));
R = rand(n);
[R,dummy] = eig( R+R' );
T = 4*[ rand(n,1) - rand(n,1) ];
M = R*S;

s = eye(n);
s(1,1) = 0.6;

for j = 1:7
  X = M*s*x+ repmat(T,1,N);

  % make a model
  m = EigModel_make( X, 'keepf', 0.95 );

  % compute the Mahalanobis distance for each point
  rr(j,:)  = EigModel_Mahalanobis( X, m, 'MogPen' ); % M-dist with residue estimate
  RR(j,:)  = EigModel_Mahalanobis( X, m ); % M-dist without residue estimate

  s = s*s;

end
