close all;
clear all;

n = 2; % dimensions

N = 100;
mu =  (rand(n,1)-rand(n,1));
a = (rand-rand)*pi/2;
a = 0;
R = [cos(a) -sin(a); sin(a) cos(a)];
S =  diag(rand(n,1)) + diag(rand(n,1));
M = R*S;
Z = gsamp( zeros(1,n), eye(n), N )' ;
Z = Z - repmat(mean(Z,2),1,N);
  
X = M*Z + repmat(mu,1,N);
eX = EigModel_make( X );
  
Y = X;
eY = EigModel_make( Y );
chm = EigModel_Chernoff( eX, eY );

% compute the "best" GMM for X and Y
niter = 100;
tol = 1E-3;
[mixm, mdlm] = GMM_make( [X Y], niter, tol);

% we know it comes from ONE distribution
mix1 = GMM_init( [X Y], 1 );
[mix1,iter,e] = GMM_fit( [X Y], mix1, niter,[],tol );
mdl1 = GMM_make_mdl( [X Y], mix1, 1, mix1.Nd, n );



