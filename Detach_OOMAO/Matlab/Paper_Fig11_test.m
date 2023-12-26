addpath tools/functions
clear all;clc;%close all

savePath = "./ComputeResults/paperRR/Fig11/";if ~exist(savePath, 'dir'), mkdir(savePath); end
matName = "r0PerformanceFig11A";
matName2 = "r0PerformanceFig11B";
FigurePath = "./figures/paper/Figure11/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureNameA = "fig_P1_rmse.pdf";

%% Test parameters
Rin = load(savePath+matName+".mat");R=Rin.Results;
Rin = load(savePath+matName2+".mat");Rn=Rin.ResultsN;

r0s = R{1}.INFO.D_R0s;
r0s = [50 45 40 35 30 25 20 15 10 5 1];
y1 = R{1}.RMSEpyr(1,:);
y2 = R{1}.RMSEdpwfs(1,:);
y3 = R{1}.RMSEdpwfs2(1,:);

e1 = R{1}.RMSEpyr(2,:);
e1 = squeeze(std(R{1}.Alldata(1,:,:),[],2))';
e2 = R{1}.RMSEdpwfs(2,:);
e2 = squeeze(std(R{1}.Alldata(2,:,:),[],2))';
e3 = R{1}.RMSEdpwfs2(2,:);
e3 = squeeze(std(R{1}.Alldata(3,:,:),[],2))';

y4  = Rn{1}.RMSEpyr(1,:);
y5  = Rn{1}.RMSEdpwfs(1,:);
y6  = Rn{1}.RMSEdpwfs2(1,:);

e4  = Rn{1}.RMSEpyr(2,:);
e5  = Rn{1}.RMSEdpwfs(2,:);
e6  = Rn{1}.RMSEdpwfs2(2,:);

lbltxt{1} = sprintf("PWFS-M%i, noiseless",R{1}.INFO.modulation);
lbltxt{2} = sprintf("PWFS-M%i, noisy",R{1}.INFO.modulation);
lbltxt{3} = sprintf("DPWFS-R1, noiseless");
lbltxt{4} = sprintf("DPWFS-R1, noisy");
lbltxt{5} = sprintf("PUPIL-R1, noiseless");
lbltxt{6} = sprintf("PUPIL-R1, noisy");

fig = figure('Color','w','Units','normalized','Position',[0.5436 0.2204 0.3903 0.4444]);
firstax = axes (fig, 'FontSize', 16); 

errorbar(r0s,y1,e1,'--r','LineWidth',1.5,'CapSize',8)
hold on
errorbar(r0s,y4,e4,':r','LineWidth',1.5,'CapSize',8)
errorbar(r0s,y2,e2,'--b','LineWidth',1.5,'CapSize',8)
errorbar(r0s,y5,e5,':b','LineWidth',1.5,'CapSize',8)
errorbar(r0s,y3,e3,'--k','LineWidth',1.5,'CapSize',8)
errorbar(r0s,y6,e6,':k','LineWidth',1.5,'CapSize',8)

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



function fig= filled_plot(r0,x,e, color_mean1, transparency)

mean_vec1=x;

error_vec1=e;%/sqrt(length(input_vecs1(:,1)));

color_error1=min(1, color_mean1+0.3);

x1=r0;
X=[x1,fliplr(x1)];
y1=mean_vec1+error_vec1/2;
y2=mean_vec1-error_vec1/2;
Y=[y1,fliplr(y2)];
hold on
fig{1}=fill(X,Y, color_error1, 'EdgeColor', 'none');
%fig{2}=plot(r0,mean_vec1, 'Color',[color_mean1, 1]);
set(fig{1},'facealpha',transparency);


end