function k = EigModel_Chernoff( ex, ey )

% k = EigModel_Chernoff( ex, ey )
% Compute the Chernoff error bound given two EigModels, ex, and ey.
% pre:
%   ex is an EigModel
%   ey is an EigModel
% post:
%   k - the Chernoff bound for the pair of EigModels


% The straight-forwad way, below, fails if the eigenmodel has fewer
% vectors than the embedding space.
%
% cx = ex.vct * diag( ex.val ) * ex.vct';
% cy = ey.vct * diag( ey.val ) * ey.vct';
% 
% dorg = ex.org - ey.org;
% 
% ncx = abs(det(cx));
% ncy = abs(det(cy));
% 
% [dummy,k] = fminsearch( @EigModel_chernoff_fmin, 0.5,[], cx, cy, dorg, ncx, ncy );
% 
% return;
% We overcome this by find the smallest space that covers BOTH eigenmodels
% and transform each eigenmodel into that space.

% the distance between the means
dorg = ex.org - ey.org;

% Compute a new spanning basis to compute the Chernoff bound
% in the smallest space possible
G = ex.vct' * ey.vct;
H = ey.vct - ex.vct * G;
g = ex.vct' * dorg;
h = dorg - ex.vct * g; % residue wrt X
nu = [H(:,sum(H.*H,1) > eps), h(:,sum(h.*h,1) > eps) ];
if ~isempty(nu)
  nu = orth( nu );
end
% make sure - errors can occur if the deflated dimension of X is less than that of Y
H = ex.vct' * nu;
nu = nu( :, sum( H.*H, 1 ) < eps );

% The reside in each gives the enery in the residue space.
% The size of the space may have changed - in fact may dissapear.
% so here compute the residue per "direction"...
resn1 = size( ex.vct, 1) - size(ex.vct,2 );
if resn1 > 0
  rpern1 = ex.residue / resn1;
else
  rpern1 = 0;
end

resn2 = size( ey.vct, 1) - size(ey.vct,2 );
if resn2 > 0
  rpern2 = ey.residue / resn2;
else
  rpern2 = 0;
end

mx = size(nu,2); % the size of the additional space over ex
my = size(ex.vct,2) + mx - size(ey.vct,2);

% the covariance matrices
% can now use diagonal form for cx
cx = diag( [ex.val',  (rpern1+eps)*ones(1,mx)] );

% but need a different form for cy
% The inclusion of the residue term, coupled with numeric error
% can make identical distributions appear to be slightly different.
% inflate the system slightly too (add eps to rpern2) because
% the dimension of the space in which cx and and cy are embedded
% can be greater than the rank of either matrix. this is thought to hapen
% the distance between the origins introduces a new set of dimensions common
% to neither.

if size(ey.vct,2) == 1
  dummy = 0;
end

Gamma = nu' * ey.vct;
D = G*diag(ey.val);
E = Gamma*diag(ey.val);
cy = [ D*G' D*Gamma'; ...
          E*G'  E*Gamma'] + (rpern2+eps)*eye(size(cx,2));

% a "norm" for the models (the detetminant of the covariance)
ncx = prod(ex.val) * (rpern1^mx);
%ncy = prod(ey.val) * (rpern2^my);
ncy = abs(det(cy)); % the errors mention above imply this is the best way

% project the distance bwteen means into the common space
dorg = [ex.vct nu]' * dorg;

% now minimise the "Chernoff" distance
[dummy,k] = fminsearch( @EigModel_Chernoff_fmin, 0.5,[], cx, cy, dorg, ncx, ncy );

return;

