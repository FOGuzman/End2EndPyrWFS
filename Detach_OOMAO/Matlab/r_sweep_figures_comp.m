clear
M0 = load("./Results/results_M0.mat");
M1 = load("./Results/results_M1.mat");
M1_5 = load("./Results/results_M1_5.mat");
M2 = load("./Results/results_M2.mat");

R0s = 8./linspace(0.4,1,10);

fig = figure('Color','w','Position',[390 352 1103 626]);
subplot(2,2,1)
hold on
errorbar(R0s,mean(M0.Pyr_zern_RMSE_v,2),std(M0.Pyr_zern_RMSE_v,[],2),'k','LineWidth',1.5)
errorbar(R0s,mean(M0.Net_zern_RMSE_v,2),std(M0.Net_zern_RMSE_v,[],2),'--k','LineWidth',1.5)
box on;grid on;xlim([R0s(end) R0s(1)]);set(gca,'XDir','reverse','FontSize',14)
xlabel("$\frac{D}{r_0}$",'interpreter','latex','FontSize',18);
ylabel("$RMSE(Z,\hat{Z})$",'interpreter','latex','FontSize',18);
title("No-modulated",'interpreter','latex','FontSize',18);
legend("Pyr","Pyr+DE",'interpreter','latex','FontSize',12)
subplot(2,2,2)
hold on
errorbar(R0s,mean(M1.Pyr_zern_RMSE_v,2),std(M1.Pyr_zern_RMSE_v,[],2),'b','LineWidth',1.5)
errorbar(R0s,mean(M1.Net_zern_RMSE_v,2),std(M1.Net_zern_RMSE_v,[],2),'--b','LineWidth',1.5)
box on;grid on;xlim([R0s(end) R0s(1)]);set(gca,'XDir','reverse','FontSize',14)
xlabel("$\frac{D}{r_0}$",'interpreter','latex','FontSize',18);
ylabel("$RMSE(Z,\hat{Z})$",'interpreter','latex','FontSize',18);
title("$\mathrm{Modulation} = 1$",'interpreter','latex','FontSize',18);
legend("Pyr","Pyr+DE",'interpreter','latex','FontSize',12)
box on;grid on
legend("Pyr","Pyr+DE")
subplot(2,2,3)
hold on
errorbar(R0s,mean(M1_5.Pyr_zern_RMSE_v,2),std(M1_5.Pyr_zern_RMSE_v,[],2),'g','LineWidth',1.5)
errorbar(R0s,mean(M1_5.Net_zern_RMSE_v,2),std(M1_5.Net_zern_RMSE_v,[],2),'--g','LineWidth',1.5)
box on;grid on;xlim([R0s(end) R0s(1)]);set(gca,'XDir','reverse','FontSize',14)
xlabel("$\frac{D}{r_0}$",'interpreter','latex','FontSize',18);
ylabel("$RMSE(Z,\hat{Z})$",'interpreter','latex','FontSize',18);
title("$\mathrm{Modulation} = 1.5$",'interpreter','latex','FontSize',18);
legend("Pyr","Pyr+DE",'interpreter','latex','FontSize',12)
box on;grid on
legend("Pyr","Pyr+DE")
subplot(2,2,4)
hold on
errorbar(R0s,mean(M2.Pyr_zern_RMSE_v,2),std(M2.Pyr_zern_RMSE_v,[],2),'m','LineWidth',1.5)
errorbar(R0s,mean(M2.Net_zern_RMSE_v,2),std(M2.Net_zern_RMSE_v,[],2),'--m','LineWidth',1.5)
box on;grid on;xlim([R0s(end) R0s(1)]);set(gca,'XDir','reverse','FontSize',14)
xlabel("$\frac{D}{r_0}$",'interpreter','latex','FontSize',18);
ylabel("$RMSE(Z,\hat{Z})$",'interpreter','latex','FontSize',18);
title("$\mathrm{Modulation} = 2$",'interpreter','latex','FontSize',18);
legend("Pyr","Pyr+DE",'interpreter','latex','FontSize',12)
box on;grid on
legend("Pyr","Pyr+DE")