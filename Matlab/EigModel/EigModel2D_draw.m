function h = EigModel2D_draw( m,colour, opt, opt2 )

% h = EigModel2D_draw( m,colour )
% Draw a TWO_DIMENSIONAL Eigenmodel (ie, has two eigenvalues),
% using colour.

held = ishold;
hold on;


if nargin == 2 % can plot a different contour, if want to
  opt = 1;
end
if nargin < 4
    opt2 = 1;
end

c = m.org;
l = opt*sqrt(m.val);
a = l(1)*m.vct(:,1);
if size(m.val,1) == 2
  b = l(2)*m.vct(:,2);
else
  b = [0;0];
end

% plot major and minor axes
if opt2
    h(1) = line( [c(1)-a(1) c(1)+a(1)], [c(2)-a(2) c(2)+a(2)] );
    h(2) = line( [c(1)-b(1) c(1)+b(1)], [c(2)-b(2) c(2)+b(2)] );
else
    h = [];
end

% Plot ellipses of one standard deviation
if size(m.val,1) == 2
  theta = 0:0.02:2*pi;
  Nt = size(theta,2);
  x = m.vct*diag(l)*[cos(theta);sin(theta)] + repmat(c,1,Nt);
  h(end+1) = plot(x(1,:),x(2,:));
end

set(h,'Color',colour);

if held == 0
  hold off; % return state of figure
end

return;
