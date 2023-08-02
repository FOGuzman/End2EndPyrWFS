addpath tools/functions
clear all;clc;close all

savePath = "./ComputeResults/paper/Fig9/";if ~exist(savePath, 'dir'), mkdir(savePath); end
matName = "Dr0_5_PerformanceFig9";
FigurePath = "./figures/paper/Figure9/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end

Resoultion = 2400;
%% A
FigureName = "ElementA.png";
fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.3465 0.6331]);
Rin = load(savePath+matName+".mat");R=Rin.Results{1};

Dr0 = R.INFO.D_r0;

rx = meshgrid(R.INFO.Readout)';
py = meshgrid(R.INFO.PhotonN);

y1 = squeeze(R.RMSEpyr(1,:,:));
y2 = squeeze(R.RMSEdpwfs(1,:,:));
y3 = squeeze(R.RMSEdpwfs2(1,:,:));

lbltxt{1} = sprintf("PWFS-M%i",R.INFO.modulation);
lbltxt{2} = sprintf("DPWFS-R1");
lbltxt{3} = sprintf("DPWFS-N1");

surf(rx,py,y1,'FaceColor', [250 0 0]/255,'FaceAlpha', 0.55,'EdgeColor','none')
hold on
surf(rx,py,y2,'FaceColor', [0 0 255]/255,'FaceAlpha', 0.55,'EdgeColor','none')
surf(rx,py,y3,'FaceColor', [0 180 0]/255,'FaceAlpha', 0.55,'EdgeColor','none')
l1 = legend(lbltxt,'FontSize',12,'position',[0.1736 0.9312 0.7432 0.0453]);
l1.Orientation = 'horizontal';

zlabel('RMSE [radians]','FontSize',16)
xl1 = xlabel('Readout noise','FontSize',16);
xl1.Rotation = 20;xl1.Position = [0.7 0.01 0.01];
yl1 = ylabel('Photon noise','FontSize',16);
yl1.Rotation = -35;yl1.Position = [-0.16 0.12 0.01];
set(gca,'FontSize',16)
box on
tt = title(" ");

exportgraphics(fig,FigurePath+FigureName,'Resolution',Resoultion)
%% B
FigureName = "ElementB.png";
fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.3465 0.6331]);
matName = "Dr0_10_PerformanceFig9";

Rin = load(savePath+matName+".mat");R=Rin.Results{1};

y1 = squeeze(R.RMSEpyr(1,:,:));
y2 = squeeze(R.RMSEdpwfs(1,:,:));
y3 = squeeze(R.RMSEdpwfs2(1,:,:));

surf(rx,py,y1,'FaceColor', [250 0 0]/255,'FaceAlpha', 0.55,'EdgeColor','none')
hold on
surf(rx,py,y2,'FaceColor', [0 0 255]/255,'FaceAlpha', 0.55,'EdgeColor','none')
surf(rx,py,y3,'FaceColor', [0 180 0]/255,'FaceAlpha', 0.55,'EdgeColor','none')
l1 = legend(lbltxt,'FontSize',12,'position',[0.1841 0.8303 0.7432 0.0453]);
l1.Orientation = 'horizontal';
l1.Visible = 'off';
zlabel('RMSE [radians]','FontSize',16)
xl1 = xlabel('Readout noise','FontSize',16);
xl1.Rotation = 20;xl1.Position = [0.7 0.01 0.01];
yl1 = ylabel('Photon noise','FontSize',16);
yl1.Rotation = -35;yl1.Position = [-0.16 0.12 0.01];
set(gca,'FontSize',16)
box on
title("(b)")
exportgraphics(fig,FigurePath+FigureName,'Resolution',Resoultion)
%% C
FigureName = "ElementC.png";
matName = "Dr0_15_PerformanceFig9";

Rin = load(savePath+matName+".mat");R=Rin.Results{1};

y1 = squeeze(R.RMSEpyr(1,:,:));
y2 = squeeze(R.RMSEdpwfs(1,:,:));
y3 = squeeze(R.RMSEdpwfs2(1,:,:));

fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.3465 0.6331]);
surf(rx,py,y1,'FaceColor', [250 0 0]/255,'FaceAlpha', 0.55,'EdgeColor','none')
hold on
surf(rx,py,y2,'FaceColor', [0 0 255]/255,'FaceAlpha', 0.55,'EdgeColor','none')
surf(rx,py,y3,'FaceColor', [0 180 0]/255,'FaceAlpha', 0.55,'EdgeColor','none')
l1 = legend(lbltxt,'FontSize',12,'position',[0.1841 0.8303 0.7432 0.0453]);
l1.Orientation = 'horizontal';
l1.Visible = 'off';
zlabel('RMSE [radians]','FontSize',16)
xl1 = xlabel('Readout noise','FontSize',16);
xl1.Rotation = 20;xl1.Position = [0.7 0.01 -0.025];
yl1 = ylabel('Photon noise','FontSize',16);
yl1.Rotation = -35;yl1.Position = [-0.16 0.12 -0.02];
set(gca,'FontSize',16)
box on
title("(c)")

exportgraphics(fig,FigurePath+FigureName,'Resolution',Resoultion)

%% D
FigureName = "ElementD.png";
matName = "Dr0_20_PerformanceFig9";

Rin = load(savePath+matName+".mat");R=Rin.Results{1};

y1 = squeeze(R.RMSEpyr(1,:,:));
y2 = squeeze(R.RMSEdpwfs(1,:,:));
y3 = squeeze(R.RMSEdpwfs2(1,:,:));

fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.3465 0.6331]);
surf(rx,py,y1,'FaceColor', [250 0 0]/255,'FaceAlpha', 0.55,'EdgeColor','none')
hold on
surf(rx,py,y2,'FaceColor', [0 0 255]/255,'FaceAlpha', 0.55,'EdgeColor','none')
surf(rx,py,y3,'FaceColor', [0 180 0]/255,'FaceAlpha', 0.55,'EdgeColor','none')
l1 = legend(lbltxt,'FontSize',12,'position',[0.1841 0.8303 0.7432 0.0453]);
l1.Orientation = 'horizontal';
l1.Visible = 'off';
zlabel('RMSE [radians]','FontSize',16)
xl1 = xlabel('Readout noise','FontSize',16);
xl1.Rotation = 20;xl1.Position = [0.7 0.01 -0.03];
yl1 = ylabel('Photon noise','FontSize',16);
yl1.Rotation = -35;yl1.Position = [-0.16 0.12 -0.024];
set(gca,'FontSize',16)
box on
title("(d)")

exportgraphics(fig,FigurePath+FigureName,'Resolution',Resoultion)