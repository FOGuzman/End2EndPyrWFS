addpath tools/functions
clear all;clc;close all

savePath = "./ComputeResults/paper/Fig7/";if ~exist(savePath, 'dir'), mkdir(savePath); end
matName = "Dr0_5_PerformanceFig7";
FigurePath = "./figures/paper/Figure7/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end

Resoultion = 1200;


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

surf(rx,py,y1,'FaceColor', [120 120 255]/255,'FaceAlpha', 0.55)
hold on
surf(rx,py,y2,'FaceColor', [100 255 100]/255,'FaceAlpha', 0.55)
surf(rx,py,y3,'FaceColor', [255 50 50]/255,'FaceAlpha', 0.55)
l1 = legend(lbltxt,'FontSize',12,'position',[0.1841 0.8303 0.7432 0.0453]);
l1.Orientation = 'horizontal';

zlabel('RMSE','FontSize',16)
xlabel('Readout noise','FontSize',16)
ylabel('Photon noise','FontSize',16)
set(gca,'FontSize',16)
box on
title("(a)")

exportgraphics(fig,FigurePath+FigureName,'Resolution',Resoultion)
%% B
FigureName = "ElementB.png";
fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.3465 0.6331]);
matName = "Dr0_10_PerformanceFig7";

Rin = load(savePath+matName+".mat");R=Rin.Results{1};

y1 = squeeze(R.RMSEpyr(1,:,:));
y2 = squeeze(R.RMSEdpwfs(1,:,:));
y3 = squeeze(R.RMSEdpwfs2(1,:,:));

surf(rx,py,y1,'FaceColor', [120 120 255]/255,'FaceAlpha', 0.55)
hold on
surf(rx,py,y2,'FaceColor', [100 255 100]/255,'FaceAlpha', 0.55)
surf(rx,py,y3,'FaceColor', [255 50 50]/255,'FaceAlpha', 0.55)
l1 = legend(lbltxt,'FontSize',12,'position',[0.1841 0.8303 0.7432 0.0453]);
l1.Orientation = 'horizontal';
zlabel('RMSE','FontSize',16)
xlabel('Readout noise','FontSize',16)
ylabel('Photon noise','FontSize',16)
set(gca,'FontSize',16)
box on
title("(b)")
exportgraphics(fig,FigurePath+FigureName,'Resolution',Resoultion)
%% C
FigureName = "ElementC.png";
matName = "Dr0_15_PerformanceFig7";

Rin = load(savePath+matName+".mat");R=Rin.Results{1};

y1 = squeeze(R.RMSEpyr(1,:,:));
y2 = squeeze(R.RMSEdpwfs(1,:,:));
y3 = squeeze(R.RMSEdpwfs2(1,:,:));

fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.3465 0.6331]);
surf(rx,py,y1,'FaceColor', [120 120 255]/255,'FaceAlpha', 0.55)
hold on
surf(rx,py,y2,'FaceColor', [100 255 100]/255,'FaceAlpha', 0.55)
surf(rx,py,y3,'FaceColor', [255 50 50]/255,'FaceAlpha', 0.55)
l1 = legend(lbltxt,'FontSize',12,'position',[0.1841 0.8303 0.7432 0.0453]);
l1.Orientation = 'horizontal';
zlabel('RMSE','FontSize',16)
xlabel('Readout noise','FontSize',16)
ylabel('Photon noise','FontSize',16)
set(gca,'FontSize',16)
box on
title("(c)")

exportgraphics(fig,FigurePath+FigureName,'Resolution',Resoultion)
