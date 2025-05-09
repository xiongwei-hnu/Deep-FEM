[N, E] = getNE();
xy = N(:,2:4);
x = xy(:, 1);
z = xy(:, 2);
y = xy(:, 3);
U = getU();
% U1 = getU1();
load('u.mat')
u_abaqus = U(:, 2:4);
% u_abaqus = u;
% u_abaqus1 = U1(:, 2:4);
d1 = 1;
d2 = 2;
d3 = 3;
f = 100;
c = u_abaqus(:, d1);
c_p = u(:, d1) / f;
l = u_abaqus(:, d2);
l_p = u(:, d2) / f;
m = u_abaqus(:, d3);
m_p = u(:, d3) / f;
load('ii.mat')
size = 26;
kuang = [100,100,600,450];
jiao = [29.653015182826028,27.46081804862591];

% ii = [];
% 
% for e = 1:length(E)
%     i = E(e,2:5);
%     index = [1 1 2 1; 2 2 3 3; 3 4 4 4];    
%     ii = [ii i(index)]; 
% end
x1 = x(ii);
y1 = y(ii);
z1 = z(ii);
% z1 = z(ii);
c1 = c(ii);
cp1 = c_p(ii);
c2 = l(ii);
cp2 = l_p(ii);
c3 = m(ii);
cp3 = m_p(ii);

figure(1);
patch(x1, y1, z1, c1, 'FaceColor','interp');
shading interp;
set(gcf,'unit','centimeters','position',[10,5,40,15]);
colormap(jet)
% hold off
view(jiao)
colorbar();
% colorbar('horiz');
xlabel('x')
ylabel('y')
zlabel('z')
% title(strcat('abaqus:u_',int2str(d1)))
title(strcat(''));
% xlabel('')
% ylabel('')
% xlim([-20 20])
% ylim([-20 20])
set(gca, 'FontSize', size);
axis equal
% saveas(gcf, './tuu/a1.png');

figure(2);
patch(x1, y1, z1, cp1, 'FaceColor','interp');
shading interp;
set(gcf,'unit','centimeters','position',[10,5,40,15]);
colormap(jet)
% hold off
view(jiao)
colorbar();
title(strcat(''));
xlabel('x')
ylabel('y')
zlabel('z')
% title(strcat('dfem:u_',int2str(d1)))
% xlim([-20, 20])
% ylim([-20, 20])
set(gca, 'FontSize', size);
axis equal
% saveas(gcf, './tuu/u1.png');

figure(3);
patch(x1, y1, z1, abs(cp1-c1), 'FaceColor','interp');
shading interp;
set(gcf,'unit','centimeters','position',[10,5,40,15]);
colormap(jet)
view(jiao)
colorbar();
title(strcat(''));
xlabel('x')
ylabel('y')
zlabel('z')
% title(strcat('error:u_',int2str(d1)))
% xlim([-20, 20])
% ylim([-20, 20])
set(gca, 'FontSize', size);
axis equal
% saveas(gcf, './tuu/e1.png');

figure(4);
patch(x1, y1, z1, c3, 'FaceColor','interp');
shading interp;
set(gcf,'unit','centimeters','position',[10,5,40,15]);
colormap(jet)
% hold off
view(jiao)
colorbar();
title(strcat(''));
xlabel('x')
ylabel('y')
zlabel('z')
% title(strcat('abaqus:u_',int2str(d2)))
% xlim([-20, 20])
% ylim([-20, 20])
set(gca, 'FontSize', size);
axis equal
% saveas(gcf, './tuu/a2.png');

figure(5);
patch(x1, y1, z1, cp3, 'FaceColor','interp');
shading interp;
set(gcf,'unit','centimeters','position',[10,5,40,15]);
colormap(jet)
% hold off
view(jiao)
colorbar();
title(strcat(''));
xlabel('x')
ylabel('y')
zlabel('z')
% title(strcat('dfem:u_',int2str(d2)))
% xlim([-20, 20])
% ylim([-20, 20])
set(gca, 'FontSize', size);
axis equal
% saveas(gcf, './tuu/u2.png');

figure(6);
patch(x1, y1, z1, abs(cp3-c3), 'FaceColor','interp');
shading interp;
set(gcf,'unit','centimeters','position',[10,5,40,15]);
colormap(jet)
view(jiao)
colorbar();
title(strcat(''));
xlabel('x')
ylabel('y')
zlabel('z')
% title(strcat('error:u_',int2str(d2)))
% xlim([-20, 20])
% ylim([-20, 20])
set(gca, 'FontSize', size);
axis equal
% saveas(gcf, './tuu/e2.png');

figure(7);
patch(x1, y1, z1, c2, 'FaceColor','interp');
shading interp;
set(gcf,'unit','centimeters','position',[10,5,40,15]);
colormap(jet)
% hold off
view(jiao)
colorbar();
title(strcat(''));
xlabel('x')
ylabel('y')
zlabel('z')
% title(strcat('abaqus:u_',int2str(d3)))
% xlim([-20, 20])
% ylim([-20, 20])
set(gca, 'FontSize', size);
axis equal
% saveas(gcf, './tuu/a3.png');

figure(8);
patch(x1, y1, z1, cp2, 'FaceColor','interp');
shading interp;
set(gcf,'unit','centimeters','position',[10,5,40,15]);
colormap(jet)
% hold off
view(jiao)
colorbar();
title(strcat(''));
xlabel('x')
ylabel('y')
zlabel('z')
% title(strcat('dfem:u_',int2str(d3)))
% xlim([-20, 20])
% ylim([-20, 20])
set(gca, 'FontSize', size);
axis equal
% saveas(gcf, './tuu/u3.png');

figure(9);
patch(x1, y1, z1, abs(cp2-c2), 'FaceColor','interp');
shading interp;
set(gcf,'unit','centimeters','position',[10,5,40,15]);
colormap(jet)
view(jiao)
colorbar();
title(strcat(''));
xlabel('x')
ylabel('y')
zlabel('z')
% title(strcat('error:u_',int2str(d3)))
% xlim([-1200, 300])
% ylim([-150, 150])
% set(gcf, 'Position', kuang);
set(gca, 'FontSize', size);
axis equal
% saveas(gcf, './tuu/e3.png');

% figure(10);
% scatter3(x, y, z, 3, 'ko', 'filled');
% set(gcf,'unit','centimeters','position',[10,5,40,15]);
% view(jiao)
% xlabel('x')
% ylabel('y')
% zlabel('z')
% title(strcat(''));
% set(gca, 'FontSize', size);
% axis equal
% grid off;
% saveas(gcf, './tu1/dian.png');

% figure(11);
% patch(x1, y1, z1, abs(cp3-c3) .* 0, 'FaceColor','interp');
% shading interp;
% set(gcf,'unit','centimeters','position',[10,5,40,15]);
% colormap(jet)
% view(jiao)
% colorbar();
% title(strcat(''));
% xlabel('x')
% ylabel('y')
% zlabel('z')
% title(strcat('error:u_',int2str(d3)))
% xlim([-1200, 300])
% ylim([-150, 150])
% set(gcf, 'Position', kuang);
% set(gca, 'FontSize', size);
% axis equal
% axis off;
% saveas(gcf, './tu1/model.png');