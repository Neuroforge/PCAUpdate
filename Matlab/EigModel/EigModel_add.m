function m3 = EigModel_add( m1, m2, method, param  )

% m3 = EigModel_add( m1, m2, method, param  )
%
% Add eigenmodels m1 and m2 to produce eigenmodel m3
% Keep eigenvectors using method specified by retmethod,
% to an extent specified by retparam.
% pre:
%  m1 is an EigModel
%  m2 is an EigModel
%  retmethod is an optional string, must be one of:
%    'keepN' - to keep N of the eigenvectors
%    'keepf' - to keep a % of the eigenenergy
%    'keept' - to keep eigenvalues above a threshold
%    'keepr' - to keep a faction of the eigenvectors
%  retparam is compulsory if retmethod is defined, prohibited otherwise.
%    It is a numerical value whose meaning depend on retmethod.
% post:
%  m3 is an eigenmodel:
% Notes:
%  For an EigModel m
%    m.org is the mean of the observations.
%    m.vct is a (n x p) matrix of eigenvectors (columns)
%	   p is decided by the <retmethod> above.
%    m.val is a (p x 1)  matrix (column vector) of eigenavlues.
%    m.N is the number of observations input to make the model.
%	 The ith eigenvalue and eigenvecvtor correspond, and the e-values
%	 are ordered by decending value.
%	 Hence, the ordering of the eigenvectors may different from that
%	 of any support vectors input, and there is no guarantee that
%	 the eigenvectors form a right-handed set.
% See also:
%  EigModel_make
%  EigModel_rank

%
% Throw exceptions
%
if ( m1.N == 0 )
	m3.N = m2.N;
	m3.org = m2.org;
	m3.vct = m2.vct;
	m3.val = m2.val;
    m3.residue = m2.residue;
	return;
end
if ( m2.N == 0 )
	m3.N = m1.N;
	m3.org = m1.org;
	m3.vct = m1.vct;
	m3.val = m1.val;
    m3.residue = m1.residue;
	return;
end

% Compute the new number of observations
m3.N = m1.N + m2.N;

if ( m3.N == 0 )
	% a "negative" eigenspace has been added to a positive, to get a null space.  
	m3.org = [];
	m3.vct = [];
	m3.val = [];
  m3.residue = 0;
	return;
end

% The body of the function follows....

% Compute the new origin
m3.org = (m1.N*m1.org + m2.N*m2.org)/m3.N;

% Store the vector between origins for later use
dorg = m1.org - m2.org;

% Compute a new spanning basis
G = m1.vct' * m2.vct;
H = m2.vct - m1.vct * G;
g = m1.vct' * dorg;
h = dorg - m1.vct * g; % residue wrt X
if ~isempty( [H(:,sum(H.*H,1) > eps), h(:,sum(h.*h,1) > eps) ] )
  nu = orth( [H(:,sum(H.*H,1) > eps), h(:,sum(h.*h,1) > eps) ] );
  % make sure - errors can occur if the deflated dimensionof X is less than that of Y
  H = m1.vct' * nu;
  nu = nu( :, sum( H.*H, 1 ) < eps );
else
  nu = zeros( size(m1.org,1), 0 );
end

% The residue in each gives the energy in the residue space.
% The size of the space may have changed - in fact may dissapear.
% so here compute the residue per "direction"...
resn1 = size( m1.vct, 1) - size(m1.vct,2 );
if resn1 > 0
  rpern1 = m1.residue / resn1;
else
  rpern1 = 0;
end

resn2 = size( m2.vct, 1) - size(m2.vct,2 );
if resn2 > 0
  rpern2 = m2.residue / resn2;
else
  rpern2 = 0;
end

% Compute the intermediate matrix as the sum of three matrices, A1,A2,A3...
% A term for the correlation of m1, use reside for error correction
[n m] = size(nu);
A1 = (m1.N/m3.N)*diag( [m1.val',  rpern1*ones(1,m)]);

% A term for the correlation of m2; project m2.vct onto the new basis
% use residue for error correction
Gamma = nu' * m2.vct;
D = G*diag(m2.val);
E = Gamma*diag(m2.val);
A2 = (m2.N/m3.N)*[ D*G' D*Gamma'; ...
                              E*G' E*Gamma'] + rpern2*eye(size(A1,1));

% A term for the difference between means
gamma = nu' * dorg;
A3 = (m1.N*m2.N)/(m3.N^2)*[g*g' g*gamma'; ...
                           gamma*g' gamma*gamma'];

% guard against rounding errors forcing imaginary values!
A = (A1+A2+A3);
A = (A+A')/2; 

% now can compute...
[m3.vct m3.val] = eig( A ); % the eigen-solution
m3.vct = [m1.vct nu]* m3.vct; % rotate the basis set into place - can fail for v.high dim data
m3.val = diag(m3.val);             % keep only the diagonal

% [R m3.val] = eig( A ); % the eigen-solution
% nu= [m1.vct nu];
% for i = 1:size(nu,2)
%   nu(i,:) = nu(i,:) * R;
% end
%  
% save temp m3 nu method param rpern1 rpern2;
% clear all;
% load temp;
% !del temp.mat; % assume MSDOS

% order the eigenvectors
[m3.val,i] = sort(m3.val); % sort  evals in ascending order
m3.val = flipud(m3.val);   % reverse their order
i = flipud(i);             % reverse order of indices too
%n = min(  max( size(m1.val,1), size(m2.val,1) ), length(m3.val) ); % max poss rank
% n = min( [m3.N, size(m1.vct,1), length(m3.val)] ); % this allows rank to fall!
%m3.val = m3.val( 1:n );       % permute eigenvalues into order, and trim
m3.vct = m3.vct( :,i );  % same for eigenvectors

if nargin == 4 % Deflate the EigModel
  n = Emodel_rank( m3.val, method, param );
  m3.val = m3.val( 1:n );
  m3.vct = m3.vct( :, 1:n );
else % make sure every eigenvalue is >= 0
  n = Emodel_rank( m3.val, 'keept', eps ); % max actual rank
  m3.val = m3.val( 1:n );
  m3.vct = m3.vct( :, 1:n );
end


resn3 = size( m3.vct,1 ) - size( m3.vct,2 );
% the add the residues per direction, and scale by the number of residue
% directions in the result.
m3.residue = resn3*( rpern1 + rpern2 );



return;
