function h = EigModel_residue( x, e )

% h = EigModel_residue( x, e )
% Return the residue vectors for the dat ax and model e.

h = EigModel_unproject( EigModel_project( x, e ) ,e) - x;

return;
