clear all; clc

%%
saveFold = "../Preconditioners/nocap/base/";
load(saveFold+"rmseResults.mat")

x = R(1).D_R0s;

y1 = R(1).meanRMSEpyr;
y2 = R(2).meanRMSEpyr;
y3 = R(3).meanRMSEpyr;
y4 = R(1).meanRMSEde;
y5 = R(2).meanRMSEde;
y6 = R(3).meanRMSEde;

lbltxt{1} = sprintf("Pyr Mod $= %i\\lambda/D_0$",R(1).modulation);
lbltxt{2} = sprintf("Pyr Mod $= %i\\lambda/D_0$",R(2).modulation);
lbltxt{3} = sprintf("Pyr Mod $= %i\\lambda/D_0$",R(3).modulation);
lbltxt{4} = sprintf("Pyr+DE Mod $= %i\\lambda/D_0$",R(1).modulation);
lbltxt{5} = sprintf("Pyr+DE Mod $= %i\\lambda/D_0$",R(2).modulation);
lbltxt{6} = sprintf("Pyr+DE Mod $= %i\\lambda/D_0$",R(3).modulation);

figure('Color','w','Position',[607 316 680 547])
% ha = tight_subplot(1,2,[.0 .06],[0.12 .03],[.01 .01]);
% axes(ha(1));
% imagesc(R(1).ExampleMeas);colormap jet;axis off
% axes(ha(2));
plot(x,y1,'--dr','LineWidth',1.5,'MarkerFaceColor','r')
hold on
plot(x,y2,'--dg','LineWidth',1.5,'MarkerFaceColor','g')
plot(x,y3,'--db','LineWidth',1.5,'MarkerFaceColor','b')
plot(x,y4,'-or','LineWidth',1.5,'MarkerFaceColor','r')
plot(x,y5,'-og','LineWidth',1.5,'MarkerFaceColor','g')
plot(x,y6,'-ob','LineWidth',1.5,'MarkerFaceColor','b')
set(gca,'XDir','reverse','FontSize',15,'TickLabelInterpreter','latex')
xlabel('$D/r_0$','interpreter','latex','FontSize',18)
ylabel('RMSE','interpreter','latex','FontSize',18)
legend(lbltxt,'interpreter','latex')