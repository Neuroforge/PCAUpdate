function x = EigModel2D_draw( m, N )

% x = EigModel2D_pntbound( m )
% x is the set of N points on the boundary of an eigenmodel

a = (2*pi)/(N-1);

theta = 0:a:2*pi;
x = m.vct*diag(sqrt(m.val))*[cos(theta);sin(theta)] + repmat(m.org,1,N);


return;
