function x = EigModel_unwhiten( y, e )
%% EigModel_whiten
% perform a whitening transform on x, given eigenmodel e.

y = y .* repmat( sqrt(e.val),1,size(y,2));
x = EigModel_unproject( y, e );


end

