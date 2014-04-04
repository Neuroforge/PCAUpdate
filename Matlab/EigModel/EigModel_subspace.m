function eout = EigModel_subspace( ein, P )

% eout = EigModel_subspace( ein, P )
% Generate a new Eigenmodel by projecting an existing one into a subspace.
% pre:
%   ein - an existing EigModel
%   P   - an orthogonal projection matrix
% post:
%  eout - the output EigModel


eout.N = ein.N; % copy the number of points
eout.org = P*ein.org; % project centres

[eout.vct,val] = eig(P*ein.vct*diag(ein.val)*ein.vct'*P'); % get the new evect,eval
[eout.val,i] = sort(diag(val)); % sort them, as required for an EigModel
eout.val = flipud(eout.val);
eout.vct = eout.vct(:,flipud(i));

i = find(diag(P == 0)); % estimate reside at energy in the "null" subspace
eout.residue = sum( ein.val(i) );

eout = EigModel_deflate( eout, 'keept', eps ); % keep only the largest evects

return;
