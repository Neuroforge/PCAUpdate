function m = EigModel_make( obs, method, param )

% m = EigModel_make( obs, method, param )
%
% Make an eigenspace model for observations
% Use the most efficient method possible -
% "efficient" meaning solve the smallest sized problem.
% The cost of this is that when solving based on inner products
% the accuracy is less that for the straight forward method.
% The inner product method, is based on
%	   L. Sirovich and M. Kirby (1987)
%	   Low-dimensional procedure for the characterization of human faces.
%	   Journal Optical Society of America, A: 4(3), 1987, 519-524.
%
% pre:
%	  obs - an (nxN) matrix; N observations each n dimensional
%	  method - an optional string, must be one of:
%		  'keepN' - to keep N of the eigenvectors
%		  'keepf' - to keep a % of the eigenenergy
%		  'keept' - to keep eigenvalues above a threshold
%		  'keepr' - to keep a faction of the eigenvectors
%	  param is compulsory if retmethod is defined, prohibited otherwise.
%		      it is a numerical value whose meaning depend on retmethod.
% post:
%	  m - an eigenmodel; a structure comprising
%	    m.N - the number of observations input to make the model.
%	    m.org - the mean of the observations.
%	    m.vct - a (n x p) matrix of eigenvectors (columns)
%		          p is decided by the <retmethod> above, and is at
%             most min(n,N).
%	    m.val - a (p x 1)  matrix (column vector) of eigenavlues.
% Notes:
%	  The ith eigenvalue and eigenvecvtor correspond, and the e-values
%	  are ordered by decending value.
%   The eigenvectors are left-handed in that m.vct'mvct = I,
%   but and there is no guarantee that they are right-handed set, for
%   m.vct*mvct' ~= I, if p < n.
%
% See also:
%	  EigModel_add


% the size of the problem
%
[n m.N] = size( obs );

if m.N == 0  % there is no data!
  m.org = [];
  m.vct = [];
  m.val = [];
  m.residue = 0;
  return;
end

if m.N == 1 % not enough data for EVD
  m.org = obs;
	m.vct = zeros(n,1);
	m.val = [0];
  m.residue = 0;
	return;
end

% use the most efficient method
%
if n <= m.N
  % straight forward method
  m.org = sum( obs, 2 ) / m.N;   % origin is the average
  obs = obs - repmat(m.org,1,m.N); % shift observations to new origin
  [m.vct,m.val] = eig( (obs * obs')/m.N );% compute EVD
  m.val = diag(m.val);
  % order the eigenvectors
  [m.val,i] = sort(m.val); % sort  evals in ascending order
  m.val = flipud(m.val);   % reverse their order
  i = flipud(i);           % reverse order of indices too
  m.val = m.val(1:n);      % keep only number of observations
  m.vct = m.vct( :, i(1:n) );
else
  % inner product method, based on
  m.org = sum( obs, 2 ) / m.N;   % origin is the average
  obs = obs - repmat(m.org,1,m.N); % shift observations to new origin
  [m.vct,m.val] = eig( (obs' * obs)/m.N ); % compute EVD of inner product
  m.val = diag(m.val);
  
  % need to keep and rotate the largest N eigenvectors
  [m.val,i] = sort(m.val); % sort  evals in ascending order
  m.val = flipud(m.val);   % reverse their order
  i = flipud(i);           % reverse order of indices too
  m.val = m.val(1:m.N);    % keep only number of observations
  m.vct = m.vct( :, i(1:m.N) );
  m.vct = obs * m.vct * diag( (m.val*m.N).^(-1/2) );%rotate
end

dummy = m.val;
if nargin == 3 % Deflate the EigModel using specified method
  n = Emodel_rank( m.val, method, param );
  m.val = m.val( 1:n );
  m.vct = m.vct( :, 1:n );
else % make sure every eigenvalue is >= eps
%   n = Emodel_rank( m.val, 'keept',  eps );
%   m.val = m.val( 1:n );
%   m.vct = m.vct( :, 1:n );
end
dummy = dummy( n+1: end );
dummy = dummy( dummy > 0 );
m.residue = sum( dummy ); % total error estimate



% if every datum input is exactly equal, then the spread is zero and ALL
% eignevectors and eigenvalues will have been removed. We can "fudge" a
% representation by including zeros vectors and zero values.
if isempty( m.val )
  m.vct = zeros(size(m.org,1),1);
  m.val = 0;
end

% can VERY occasioanlly end up with a ZERO) imaginary field
% (I do not know why)
m.vct = real( m.vct );
m.val = abs( m.val );

return;
