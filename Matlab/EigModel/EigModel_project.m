function u = EigModel_project( x, model )

% u = EigModel_project( x, model )
% Project the data in x into the eigenmodel to get u.
% pre:
%   x - data to be projected
%   model - an EigModel
% post:
%   u - the projected data
% see also
%   EigModel_unproject

u = model.vct' * (x - repmat(model.org,1,size(x,2)));

return;

