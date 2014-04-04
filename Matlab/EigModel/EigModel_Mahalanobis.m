function d = EigModel_Mahalanobis( x, m, opt )

% EigModel_Mahalanobis - compute the Mahalanobis distance of points x
%			 to a model m.
% Usage:
% 	d = EigModel_Mahalanobis( x, m, opt )
% pre:
%	x is a (n x N) matrix: N points, each n-dimensional
%	m is an eigenmodel
%	opt is an optional argument. If passed with value
%		'MogPen'
%	the Mahalanobis distance is estimated using a method due
%	to Mogghaden and Pentland.  It is applicable in cases where
%	the Eigmodel space a subspace of the embedding space.
% post:
%	d is a (1 x N) matrix: N distances, one for each point
% depends on:
%	-
% notes:
%	-
% see also
%	EigModel_make

% N is the number of points
% n is the dimension of the space
[n N] = size(x);

if m.N == 0
  % the Mahalanobis distance is undefined for an empty model
  d = NaN*ones(1,N);
  return;
end

if m.N == 1
  % Only one element in the EigModel, so val = 0.
  %  We default to a spherical Gaussian with a tiny standard deviation.
  u = x - repmat(m.org,1,N);
  d = sum( u.*u, 1 ) / eps^2;
  return;
end

% Standard body of function...

% project the data into the Eigenspace
u = m.vct' * (x - repmat(m.org,1,N));

% compute Mahalanobis distance _in_ Eigenspace
d = sum( (u.*u) ./ (repmat(m.val,1,N)), 1 );

% can  we do better? and does the user request we try
D = size( m.vct,1) - size( m.vct,2); % dimension of null space
if  D > 0 % can do better by trying to account for the null space
  if nargin == 3
    if strcmp(opt,'MogPen') % user requests we try
      if m.residue > 0
        aval = m.residue / D ; % average variance per null vector
      else 
        aval = 1/(eps^2); % pretend a tiny variance per null vector
      end
      
      u = m.vct*u + repmat( m.org,1,N ) - x; % error vector
      d = d + sum( (u.*u) ./ repmat( aval, n, N ) );
      
    end
  end
end

return
