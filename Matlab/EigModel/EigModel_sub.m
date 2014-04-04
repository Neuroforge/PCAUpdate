function Msub = EigModel_sub( Mell, Mring )

% Msub = EigModel_sub( Mell, Mring )

Mring.N = -Mring.N;
Msub = EigModel_add( Mell, Mring );

return;


