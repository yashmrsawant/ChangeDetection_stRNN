clear;
close all;
clc;
load('test_data_20l_off_at_11.mat');

%% Deviation loss in log scale: 
%  after swithing off EE, II, EI and IE neurons only at t=11
yy = [];
figure;
x=1:20; y=avg_loss_n_m_; xx = 1:.1:20;
for i1 = 1:263 yy(i1,:) = pchip(x,y(i1,:),xx); end
yy = max(yy, 0);

shadedErrorBar(xx, log(mean(yy)), log(std(yy))/sqrt(263), 'lineprops', '-c'); hold on;
scatter(x(1:1:end), log(mean(y(:,1:1:end))), 'o', 'c'); hold on;

x=1:20; y=avg_loss_ii_m_; xx = 1:.1:20;
for i1 = 1:263 yy(i1,:) = pchip(x,y(i1,:),xx); end
yy = max(yy, 0);

shadedErrorBar(xx, log(mean(yy)), log(std(yy))/sqrt(263), 'lineprops', '-k'); hold on;
scatter(x(1:1:end), log(mean(y(:,1:1:end))), 'o', 'k'); hold on;

x=1:20; y=avg_loss_ee_m_; xx = 1:.1:20;
for i1 = 1:263 yy(i1,:) = pchip(x,y(i1,:),xx); end
yy = max(yy, 0);

shadedErrorBar(xx, log(mean(yy)), log(std(yy))/sqrt(263), 'lineprops', '-g'); hold on;
scatter(x(1:1:end), log(mean(y(:,1:1:end))), 'o', 'g'); hold on;

x=1:20; y=avg_loss_ei_m_; xx = 1:.1:20;
for i1 = 1:263 yy(i1,:) = pchip(x,y(i1,:),xx); end
yy = max(yy, 0);

shadedErrorBar(xx, log(mean(yy)), log(std(yy))/sqrt(263), 'lineprops', '-r'); hold on;
scatter(x(1:1:end), log(mean(y(:,1:1:end))), 'o', 'r'); hold on;

x=1:20; y=avg_loss_ie_m_; xx = 1:.1:20;
for i1 = 1:263 yy(i1,:) = pchip(x,y(i1,:),xx); end
yy = max(yy, 0);

shadedErrorBar(xx, log(mean(yy)), log(std(yy))/sqrt(263), 'lineprops', '-b'); hold on;
scatter(x(1:1:end), log(mean(y(:,1:1:end))), 'o', 'b'); hold on;
xlabel('time'); ylabel('Mean error (L1, log)');
legend({'normal','','ii','','ee','','ei','','ie',''})

%% Deviation loss without log scale
figure;
x=1:20; y=avg_loss_n_m_; xx = 1:.1:20;
for i1 = 1:263 yy(i1,:) = spline(x,y(i1,:),xx); end
yy = max(yy, 0);

shadedErrorBar(xx, (mean(yy)), (std(yy)/sqrt(100)), 'lineprops', '-c'); hold on;
scatter(x(1:1:end), (mean(y(:,1:1:end))), 'o', 'c'); hold on;

x=1:20; y=avg_loss_ii_m_; xx = 1:.1:20;
for i1 = 1:263 yy(i1,:) = spline(x,y(i1,:),xx); end
yy = max(yy, 0);

shadedErrorBar(xx, (mean(yy)), (std(yy)/sqrt(100)), 'lineprops', '-k'); hold on;
scatter(x(1:1:end), (mean(y(:,1:1:end))), 'o', 'k'); hold on;

x=1:20; y=avg_loss_ee_m_; xx = 1:.1:20;
for i1 = 1:263 yy(i1,:) = spline(x,y(i1,:),xx); end
yy = max(yy, 0);

shadedErrorBar(xx, mean(yy), std(yy)/sqrt(100), 'lineprops', '-g'); hold on;
scatter(x(1:1:end), mean(y(:,1:1:end)), 'o', 'g'); hold on;

x=1:20; y=avg_loss_ei_m_; xx = 1:.1:20;
for i1 = 1:263 yy(i1,:) = spline(x,y(i1,:),xx); end
yy = max(yy, 0);

shadedErrorBar(xx, mean(yy), std(yy)/sqrt(100), 'lineprops', '-r'); hold on;
scatter(x(1:1:end), mean(y(:,1:1:end)), 'o', 'r'); hold on;

x=1:20; y=avg_loss_ie_m_; xx = 1:.1:20;
for i1 = 1:263 yy(i1,:) = spline(x,y(i1,:),xx); end
yy = max(yy, 0);

shadedErrorBar(xx, mean(yy), std(yy)/sqrt(100), 'lineprops', '-b'); hold on;
scatter(x(1:1:end), mean(y(:,1:1:end)), 'o', 'b'); hold on;
xlabel('time'); ylabel('Mean error (L1)');
legend({'normal','','ii','','ee','','ei','','ie',''})



