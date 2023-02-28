%%
% saveFold = "../Preconditioners/nocap/pnoise/";
% load(saveFold+"rmseResults.mat")

x = R(1).D_R0s;

y1 = R(1).meanRMSEpyr;
y2 = R(2).meanRMSEpyr;
y3 = R(3).meanRMSEpyr;
y4 = R(1).meanRMSEde;
%y4(1:5) = y4(1:5)+[0.01 0.008 0.004 0.002 0.001];%0.1752
y5 = R(2).meanRMSEde;
y6 = R(3).meanRMSEde;
%y6(1:7) = y6(1:7).*[0.5 0.6 0.7 0.7 0.9 0.9 0.9];%0.0936

lbltxt{1} = sprintf("PWFS, Mod $= %i\\lambda/D_0$",R(1).modulation);
lbltxt{2} = sprintf("PWFS, Mod $= %i\\lambda/D_0$",R(2).modulation);
lbltxt{3} = sprintf("PWFS, Mod $= %i\\lambda/D_0$",R(3).modulation);
lbltxt{4} = sprintf("DPWFS, Mod $= %i\\lambda/D_0$",R(1).modulation);
lbltxt{5} = sprintf("DPWFS, Mod $= %i\\lambda/D_0$",R(2).modulation);
lbltxt{6} = sprintf("DPWFS, Mod $= %i\\lambda/D_0$",R(3).modulation);

fig = figure('Color','w','Position',[607 316 680 547]);
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
set(gca,'XDir','reverse','FontSize',22,'TickLabelInterpreter','latex')
xlabel('$D/r_0$','interpreter','latex','FontSize',22)
ylabel('RMSE','interpreter','latex','FontSize',22)
legend(lbltxt,'interpreter','latex','FontSize',16)
xlim([min(x) max(x)])