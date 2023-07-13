addpath tools/functions
clear all;clc;close all

savePath = "./ComputeResults/paper/Fig8/";if ~exist(savePath, 'dir'), mkdir(savePath); end
matName = "r0PerformanceFig8A";
FigurePath = "./figures/paper/Figure8/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureNameA = "ElementA.pdf";

%% Test parameters
Rin = load(savePath+matName+".mat");R=Rin.Results;

r0s = R{1}.INFO.D_R0s;
y1 = R{1}.RMSEpyr(1,:);
y2 = R{2}.RMSEpyr(1,:);
y5 = R{1}.RMSEdpwfs(1,:);
y6 = R{1}.RMSEdpwfs2(1,:);
y7 = R{1}.RMSEdpwfs3(1,:);
y8 = R{1}.RMSEdpwfs4(1,:);

lbltxt{1} = sprintf("PWFS-M%i",R{1}.INFO.modulation);
lbltxt{2} = sprintf("PWFS-M%i",R{2}.INFO.modulation);
lbltxt{3} = sprintf("DPWFS-R1");
lbltxt{4} = sprintf("DPWFS-N1");
lbltxt{5} = sprintf("PUPIL-R1");
lbltxt{6} = sprintf("PUPIL-N1");

fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.4427 0.6331]);

plot(r0s,y1,'--dr','LineWidth',1.5,'MarkerFaceColor','r')
hold on
plot(r0s,y2,'--dm','LineWidth',1.5,'MarkerFaceColor','m')
plot(r0s,y5,'-r','LineWidth',1.5,'MarkerFaceColor','r')
plot(r0s,y6,'-g','LineWidth',1.5,'MarkerFaceColor','g')
plot(r0s,y7,'-b','LineWidth',1.5,'MarkerFaceColor','b')
plot(r0s,y8,'-m','LineWidth',1.5,'MarkerFaceColor','m')
set(gca,'XDir','reverse','FontSize',20)
xlabel('$D/r_0$','interpreter','latex','FontSize',22)
ylabel('RMSE','FontSize',22)
legend(lbltxt,'FontSize',19)
xlim([min(r0s) max(r0s)])

exportgraphics(fig,FigurePath+FigureNameA)


