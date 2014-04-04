function x = EigModel_unproject( u, model )

% x = EigModel_project( u, model )
% Un-project the data u from the eigenmodel to APPROXIMATE x
% pre:
%   u - data to be unprojected
%   model - an EigModel
% post:
%   x - the unprojected data
% see also
%   EigModel_project

x = model.vct * u + repmat(model.org,1,size(u,2));

return;
