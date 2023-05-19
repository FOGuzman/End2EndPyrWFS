clear
load ./Results/results_M1_5.mat

R0s = linspace(0.4,1,10);
z1 = std(Pyr_zern_RMSE_v,0,2)';
y1 = mean(Pyr_zern_RMSE_v,2)';
z2 = std(Net_zern_RMSE_v,0,2)';
y2 = mean(Net_zern_RMSE_v,2)';

figure('Color','w','Position',[680 558 1218 420]),
subplot(1,2,1)
hold on
plot(R0s,y1,'-sk','LineWidth',1.6)
plot(R0s,y2,'-sr','LineWidth',1.6)
grid on;box on
xlabel('$r_0$','interpreter','latex','FontSize',15)
xlim([0.1 R0s(10)])
ylabel('RMSE','interpreter','latex','FontSize',15)
title('Zernike mean RMSE','interpreter','latex','FontSize',18)
l = legend('Traditional Pyramid','Preconditioned Pyramid','interpreter','latex','FontSize',15);
l.Location = 'southwest';

subplot(1,2,2)
hold on
plot(R0s,z1,'-sk','LineWidth',1.6)
plot(R0s,z2,'-sr','LineWidth',1.6)
grid on;box on
xlabel('$r_0$','interpreter','latex','FontSize',15)
xlim([0.1 R0s(10)])
ylabel('RMSE','interpreter','latex','FontSize',15)
title('Zernike std RMSE','interpreter','latex','FontSize',18)




z1 = std(Pyr_phase_RMSE_v,0,2)';
y1 = mean(Pyr_phase_RMSE_v,2)';
z2 = std(Net_phase_RMSE_v,0,2)';
y2 = mean(Net_phase_RMSE_v,2)';



figure('Color','w','Position',[680 558 1218 420]),
subplot(1,2,1)
hold on
plot(R0s,y1,'-sk','LineWidth',1.6)
plot(R0s,y2,'-sr','LineWidth',1.6)
grid on;box on
xlabel('$r_0$','interpreter','latex','FontSize',15)
xlim([0.1 R0s(10)])
ylabel('RMSE','interpreter','latex','FontSize',15)
title('Phase mean RMSE','interpreter','latex','FontSize',18)
l = legend('Traditional Pyramid','Preconditioned Pyramid','interpreter','latex','FontSize',15);
l.Location = 'southwest';

subplot(1,2,2)
hold on
plot(R0s,z1,'-sk','LineWidth',1.6)
plot(R0s,z2,'-sr','LineWidth',1.6)
grid on;box on
xlabel('$r_0$','interpreter','latex','FontSize',15)
xlim([0.1 R0s(10)])
ylabel('RMSE','interpreter','latex','FontSize',15)
title('Phase std RMSE','interpreter','latex','FontSize',18)


