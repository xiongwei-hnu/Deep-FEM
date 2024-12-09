[N, E] = getNE();
xy = N(:,2:3);
x = xy(:, 1);
y = xy(:, 2);
% z = xy(:, 3);
U = getU();
% U1 = getU1();
load('u.mat')
u_abaqus = U(:, 2:3) * 1;
% u_abaqus = u;
% u_abaqus1 = U1(:, 2:4);
d = 1;
p = 2;
% u = u ./ 10;
c = u_abaqus(:, d);
c_p = u(:, d);
l = u_abaqus(:, p);
l_p = u(:, p);
% load('ii.mat')
sizes = [100 100 300 500];
size = 18;
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
xlim([-20 20])
ylim([-20 20])
axis equal
set(gcf, 'Position', sizes);
set(gca, 'FontSize', size);
% saveas(gcf, './tu/a1.png');
% saveas(gcf, './tu m1/ea1.png');

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
xlim([-20, 20])
ylim([-20, 20])
axis equal
set(gcf, 'Position', sizes);
set(gca, 'FontSize', size);
% saveas(gcf, './tu/pu1.png');

figure(3);
patch(x1, y1, abs(cp1-c1), 'FaceColor','interp');
shading interp;
colormap(jet)
colorbar();
% xlabel('x')
% ylabel('y')
% title(strcat('error:u_',int2str(d)))
xlim([-20, 20])
ylim([-20, 20])
axis equal
set(gcf, 'Position', sizes);
set(gca, 'FontSize', size);
% saveas(gcf, './tu/pe1.png');

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
xlim([-20, 20])
ylim([-20, 20])
axis equal
set(gcf, 'Position', sizes);
set(gca, 'FontSize', size);
saveas(gcf, './tu/a2.png');
% saveas(gcf, './tu m1/ea2.png');

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
xlim([-20, 20])
ylim([-20, 20])
axis equal
set(gcf, 'Position', sizes);
set(gca, 'FontSize', size);
% saveas(gcf, './tu/pu2.png');

figure(6);
patch(x1, y1, abs(cp2-c2), 'FaceColor','interp');
shading interp;
colormap(jet)
colorbar();
% xlabel('x')
% ylabel('y')
% title(strcat('error:u_',int2str(p)))
xlim([-20, 20])
ylim([-20, 20])
axis equal
set(gcf, 'Position', sizes);
set(gca, 'FontSize', size);
% saveas(gcf, './tu/pe2.png');