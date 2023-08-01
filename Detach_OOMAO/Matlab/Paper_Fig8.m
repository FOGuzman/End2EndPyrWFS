addpath tools/functions
clear all;clc;close all

savePath = "./ComputeResults/paper/Fig8/";if ~exist(savePath, 'dir'), mkdir(savePath); end
matName = "r0PerformanceFig8A";
matName2 = "r0PerformanceFig8B";
FigurePath = "./figures/paper/Figure8/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureNameA = "fig_P1_rmse.pdf";

%% Test parameters
Rin = load(savePath+matName+".mat");R=Rin.Results;
Rin = load(savePath+matName2+".mat");Rn=Rin.ResultsN;

r0s = R{1}.INFO.D_R0s;
r0s = [50 45 40 35 30 25 20 15 10 5 1];
y1 = R{1}.RMSEpyr(1,:);
y2 = R{1}.RMSEdpwfs(1,:);
y3 = R{1}.RMSEdpwfs2(1,:);
y4 = R{1}.RMSEdpwfs3(1,:);
y5 = R{1}.RMSEdpwfs4(1,:);

y6  = Rn{1}.RMSEpyr(1,:);
y7  = Rn{1}.RMSEdpwfs(1,:);
y8  = Rn{1}.RMSEdpwfs2(1,:);
y9  = Rn{1}.RMSEdpwfs3(1,:);
y10 = Rn{1}.RMSEdpwfs4(1,:);

lbltxt{1} = sprintf("PWFS-M%i, noiseless",R{1}.INFO.modulation);
lbltxt{2} = sprintf("PWFS-M%i, noisy",R{1}.INFO.modulation);
lbltxt{3} = sprintf("DPWFS-R1, noiseless");
lbltxt{4} = sprintf("DPWFS-R1, noisy");
lbltxt{5} = sprintf("PUPIL-R1, noiseless");
lbltxt{6} = sprintf("PUPIL-R1, noisy");

fig = figure('Color','w','Units','normalized','Position',[0.5436 0.2204 0.4427 0.4120]);
firstax = axes (fig, 'FontSize', 16); 

plot(r0s,y1,'-r','LineWidth',1.5)
hold on
plot(r0s,y6,'--r','LineWidth',1.5)
plot(r0s,y2,'-b','LineWidth',1.5)
plot(r0s,y7,'--b','LineWidth',1.5)
plot(r0s,y4,'-k','LineWidth',1.5)
plot(r0s,y9,'--k','LineWidth',1.5)

set(gca,'XDir','reverse','FontSize',16,'XTick',fliplr(r0s),'LineWidth',.8)
xlabel('$D/r_0$','interpreter','latex','FontSize',16)
ylabel('RMSE [radians]','FontSize',16)
leg1 = legend(lbltxt,'FontSize',12);
leg1.Orientation = 'horizontal';leg1.Orientation = 'vertical';
xlim([min(r0s) max(r0s)])
ylim([0 0.43])
l1.Orientation = 'horizontal';l1.Orientation = 'vertical';
grid on


% secondax = copyobj(firstax, fig);
% delete( get(secondax, 'Children'))
% H1 = plot(1, '-k', 'LineWidth', 2, 'Parent', secondax, 'Visible', 'on');
% H2 = plot(1, '--k', 'LineWidth', 2, 'Parent', secondax, 'Visible', 'on');
% set(secondax,'visible','off') 
% l2 = legend (secondax, [H1 H2], 'noiseless', 'noise','position',[0.3796 0.7358 0.2465 0.1902]);
% l2.Orientation = 'horizontal';l2.Orientation = 'vertical';




exportgraphics(fig,FigurePath+FigureNameA)


