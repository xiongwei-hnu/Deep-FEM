[N, E] = getNE();
xy = N(:,2:3);
x = xy(:, 1);
y = xy(:, 2);
% z = xy(:, 3);
U = getU();
% U1 = getU1();
load('u1.mat')
u_abaqus = U(:, 2:3);
% u_abaqus = u;
% u_abaqus1 = U1(:, 2:4);
d = 1;
p = 2;
% u = u ./ 1000;
c = u_abaqus(:, d);
c_p = u(:, d);
l = u_abaqus(:, p);
l_p = u(:, p);
% load('ii.mat')

size = 42;
kuang = [150,100,1000,600];
ii = [];

for e = 1:length(E)
    i = E(e,2:5);
    index = [1; 2; 3; 4];    
    ii = [ii i']; 
end
x1 = x(ii);
y1 = y(ii);
% z1 = z(ii);
c1 = c(ii);
cp1 = c_p(ii);
c2 = l(ii);
cp2 = l_p(ii);
figure(1);
patch(x1, y1, c1, 'FaceColor','interp');
shading interp;
% set(gcf,'unit','centimeters','position',[10,5,30,15]);
colormap(jet)
% hold off
% view([1,0,0])
colorbar();
% colorbar('horiz');
% xlabel('x')
% ylabel('y')
% zlabel('z')
% title(strcat('abaqus:u_',int2str(d)))
xlim([-40 40])
ylim([-20 20])
% axis equal
set(gcf, 'Position', kuang);
set(gca, 'FontSize', size);
% saveas(gcf, './tu1/a1.png');

figure(2);
patch(x1, y1, cp1, 'FaceColor','interp');
shading interp;
% set(gcf,'unit','centimeters','position',[10,5,30,15]);
colormap(jet)
% hold off
% view([1,0,0])
colorbar();
% xlabel('x')
% ylabel('y')
% zlabel('z')
% title(strcat('dfem:u_',int2str(d)))
xlim([-40, 40])
ylim([-20, 20])
% axis equal
set(gcf, 'Position', kuang);
set(gca, 'FontSize', size);
% saveas(gcf, './tu1/pu1.png');

figure(3);
patch(x1, y1, abs(cp1-c1), 'FaceColor','interp');
shading interp;
colormap(jet)
colorbar();
% xlabel('x')
% ylabel('y')
% title(strcat('error:u_',int2str(d)))
xlim([-40, 40])
ylim([-20, 20])
% axis equal
set(gcf, 'Position', kuang);
set(gca, 'FontSize', size);
% saveas(gcf, './tu1/pe1.png');

figure(4);
patch(x1, y1, c2, 'FaceColor','interp');
shading interp;
% set(gcf,'unit','centimeters','position',[10,5,30,15]);
colormap(jet)
% hold off
% view([1,0,0])
colorbar();
% xlabel('x')
% ylabel('y')
% zlabel('z')
% title(strcat('abaqus:u_',int2str(p)))
xlim([-40, 40])
ylim([-20, 20])
% axis equal
set(gcf, 'Position', kuang);
set(gca, 'FontSize', size);
% saveas(gcf, './tu1/a2.png');

figure(5);
patch(x1, y1, cp2, 'FaceColor','interp');
shading interp;
% set(gcf,'unit','centimeters','position',[10,5,30,15]);
colormap(jet)
% hold off
% view([1,0,0])
colorbar();
% xlabel('x')
% ylabel('y')
% zlabel('z')
% title(strcat('dfem:u_',int2str(p)))
xlim([-40, 40])
ylim([-20, 20])
% axis equal
set(gcf, 'Position', kuang);
set(gca, 'FontSize', size);
% saveas(gcf, './tu1/pu2.png');

figure(6);
patch(x1, y1, abs(cp2-c2), 'FaceColor','interp');
shading interp;
colormap(jet)
colorbar();
% xlabel('x')
% ylabel('y')
% title(strcat('error:u_',int2str(p)))
xlim([-40, 40])
ylim([-20, 20])
% axis equal
set(gcf, 'Position', kuang);
set(gca, 'FontSize', size);
% saveas(gcf, './tu1/pe2.png');

% e = abs(u_abaqus - u);
% e1 = e(:,1);
% e2 = e(:,2);
% e3 = e(:,3);
% max(max(e1));
% mean(mean(e1));
% max(max(e2));
% mean(mean(e2));