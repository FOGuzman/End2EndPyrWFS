addpath tools/functions
clear all;clc;close all

savePath = "./ComputeResults/paper/Fig8/";if ~exist(savePath, 'dir'), mkdir(savePath); end
matName = "r0PerformanceFig8A";
matName2 = "r0PerformanceFig8B";
FigurePath = "./figures/paper/Figure8/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureNameA = "ElementA.pdf";

%% Test parameters
Rin = load(savePath+matName+".mat");R=Rin.Results;
Rin = load(savePath+matName2+".mat");Rn=Rin.ResultsN;

r0s = R{1}.INFO.D_R0s;
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

lbltxt{1} = sprintf("PWFS-M%i",R{1}.INFO.modulation);
lbltxt{2} = sprintf("DPWFS-R1");
%lbltxt{3} = sprintf("DPWFS-N1");
lbltxt{3} = sprintf("PUPIL-R1");
%lbltxt{5} = sprintf("PUPIL-N1");

fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.4427 0.6331]);
firstax = axes (fig, 'FontSize', 16); 

plot(r0s,y1,'-r','LineWidth',1.5,'MarkerFaceColor','r')
hold on
plot(r0s,y2,'-g','LineWidth',1.5,'MarkerFaceColor','g')
%plot(r0s,y3,'-b','LineWidth',1.5,'MarkerFaceColor','b')
plot(r0s,y4,'-b','LineWidth',1.5,'MarkerFaceColor','b')
%plot(r0s,y5,'-y','LineWidth',1.5,'MarkerFaceColor','y')

plot(r0s,y6,'--r','LineWidth',1.5,'MarkerFaceColor','r')
plot(r0s,y7,'--g','LineWidth',1.5,'MarkerFaceColor','g')
%plot(r0s,y8,'--b','LineWidth',1.5,'MarkerFaceColor','b')
plot(r0s,y9,'--b','LineWidth',1.5,'MarkerFaceColor','b')
%plot(r0s,y10,'--y','LineWidth',1.5,'MarkerFaceColor','y')

set(gca,'XDir','reverse','FontSize',20)
xlabel('$D/r_0$','interpreter','latex','FontSize',22)
ylabel('RMSE','FontSize',22)
l1 = legend(lbltxt,'FontSize',19,'position',[0.6340 0.6289 0.2703 0.2961]);
xlim([min(r0s) max(r0s)])




secondax = copyobj(firstax, fig);
delete( get(secondax, 'Children'))
H1 = plot(1, '-k', 'LineWidth', 2, 'Parent', secondax, 'Visible', 'on');
H2 = plot(1, '--k', 'LineWidth', 2, 'Parent', secondax, 'Visible', 'on');
set(secondax,'visible','off') 
l2 = legend (secondax, [H1 H2], 'noiseless', 'noise','position',[0.6811 0.5102 0.2232 0.1193]);





exportgraphics(fig,FigurePath+FigureNameA)


