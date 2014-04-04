function M = EigModel_ellipse( e )

% M = EigModel_ellipse( eig )
% turn eigmodel into an ellipse

C = inv(e.vct * diag( e.val ) * e.vct');
m = -C * e.org;
M = [ C m; m' -e.org' * m ];

return;

