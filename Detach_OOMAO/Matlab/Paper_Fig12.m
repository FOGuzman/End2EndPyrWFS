%% Calibration 0.1
clear all; close all

savePath = "./ComputeResults/paper/Fig12/";if ~exist(savePath, 'dir'), mkdir(savePath); end
matName = "A_0_25";
FigurePath = "./figures/paper/Figure12/";if ~exist(FigurePath, 'dir'), mkdir(FigurePath); end
FigureName = "fig_experimental_RMSE.pdf";


load(savePath+matName + ".mat")
r0 = [1,5,10,15,20,25,30,35,40,45,50];
lbltxt{1} = sprintf("PWFS-M0");
lbltxt{2} = sprintf("PWFS-M2");
lbltxt{3} = sprintf("DPWFS-N1");

c1 = [0.6 0.0 0.0]; %0.6 0.0 0.0
c2 = [0.7 0.7 0.0]; %0.0 0.5 0.5
c3 = [0.0 0.5 0.0];  %1.0 1.0 0.0
tt = 0.6;


fig = figure('Color','w','Units','normalized','Position',[0.5436 0.1528 0.4427 0.5250]);
filled_plot(r0,err0,c1,err2, c2, err4, c3, tt)
ax = gca;
set(ax,'XDir','reverse','FontSize',16)
xlabel('$D/r_0$','interpreter','latex','FontSize',20)
ylabel('RMSE [radians]','FontSize',16)
legend(lbltxt,'FontSize',16,'Location','southwest')
xlim([1 35]);box on
grid on
axes('position',[.485 .6 .4 .3], 'NextPlot', 'add')

filled_plot(r0,err0,c1,err2, c2, err4, c3, tt)
ax = gca;
set(ax,'XDir','reverse','FontSize',12)
xlim([1 5]);box on
grid on

exportgraphics(fig,FigurePath+FigureName)








%% Function

function fig= filled_plot(r0,input_vecs1, color_mean1, input_vecs2, color_mean2, input_vecs3, color_mean3, transparency)

mean_vec1=mean(input_vecs1);
mean_vec2=mean(input_vecs2);
mean_vec3=mean(input_vecs3);

error_vec1=std(input_vecs1);%/sqrt(length(input_vecs1(:,1)));
error_vec2=std(input_vecs2);%/sqrt(length(input_vecs2(:,1)));
error_vec3=std(input_vecs3);%/sqrt(length(input_vecs2(:,1)));

color_error1=min(1, color_mean1+0.3);
color_error2=min(1, color_mean2+0.3);
color_error3=min(1, color_mean3+0.3);

x1=r0;
x2=r0;
x3=r0;
X1=[x1,fliplr(x1)];
X2=[x2,fliplr(x2)];
X3=[x3,fliplr(x3)];
y1=mean_vec1+error_vec1/2;
y3=mean_vec2+error_vec2/2;
y2=mean_vec1-error_vec1/2;
y4=mean_vec2-error_vec2/2;
y5=mean_vec3+error_vec3/2;
y6=mean_vec3-error_vec3/2;
Y1=[y1,fliplr(y2)];
Y2=[y3,fliplr(y4)];
Y3=[y5,fliplr(y6)];
hold on
fig{1}=fill(X1,Y1, color_error1, 'EdgeColor', 'none');
%fig{2}=plot(r0,mean_vec1, 'Color',[color_mean1, 1]);
fig{3}=fill(X2,Y2, color_error2, 'EdgeColor', 'none');
%fig{4}=plot(r0,mean_vec2, 'Color',[color_mean2, 1]);
fig{5}=fill(X3,Y3, color_error3, 'EdgeColor', 'none');
%fig{6}=plot(r0,mean_vec3, 'Color',[color_mean3, 1]);
set(fig{1},'facealpha',transparency(1));
set(fig{3},'facealpha',transparency(1));
set(fig{5},'facealpha',transparency(1));
hold off

end