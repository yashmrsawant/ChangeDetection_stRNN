clc;
close all;
clear;
%% Load fixation statistics 
load('human_stats.mat');
load('model_stats_stRNN_120fixations.mat');
mean_no_1_m_log = (mean_no_1_m);  mean_no_2_m_log = (mean_no_2_m);
mean_no_1_h_log = (mean_no_1_h);  mean_no_2_h_log = (mean_no_2_h);
std_dev_1_m_log = (std_dev_1_m/sqrt(80)); std_dev_2_m_log = (std_dev_2_m/sqrt(80));
std_dev_1_h_log = (std_dev_1_h/sqrt(40)); std_dev_2_h_log = (std_dev_2_h/sqrt(40));

%% Plot correlation curves
figure;
subplot(1, 2, 1);
errorbarxy(mean_no_1_m_log, mean_no_1_h_log, std_dev_1_m_log, std_dev_1_h_log, {'ko', 'b', 'r'}); grid on;
hold on; plot(linspace(10,70,100),linspace(0,100,100),'--k');
[RHO_1,PVAL_1] = corr(mean_no_1_m_log, mean_no_1_h_log,'Type','Pearson');
xlabel('Model');
ylabel('Human');
title(strcat('# of fixations, ', 'r = ', num2str(RHO_1), ', p=', num2str(PVAL_1)));
axis square;
subplot(1, 2, 2);
errorbarxy(mean_no_2_m_log, mean_no_2_h_log, std_dev_2_m_log, std_dev_2_h_log, {'ko', 'b', 'r'}); grid on;
hold on; plot(linspace(0,20000,100),linspace(0,14000,100),'--k');
[RHO_2,PVAL_2] = corr(mean_no_2_m_log, mean_no_2_h_log,'Type','Pearson');
title(strcat('Dist. traveled, ', 'r = ', num2str(RHO_2), ', p = ', num2str(PVAL_2)));
axis square;
xlabel('Model');
ylabel('Human');
sgtitle("With change map (+ stRNN output)");

% rho = corr(mean_no_1_m, mean_no_1_h, 'Type','Spearman');
[RHO_1,PVAL_1] = corr(mean_no_1_m_log, mean_no_1_h_log,'Type','Pearson');
[RHO_2,PVAL_2] = corr(mean_no_2_m_log, mean_no_2_h_log,'Type','Pearson');

[RHO_1_,PVAL_1_] = corr(mean_no_1_m_log, mean_no_1_h_log,'Type','Spearman');
[RHO_2_,PVAL_2_] = corr(mean_no_2_m_log, mean_no_2_h_log,'Type','Spearman');

corr(mean_no_1_m_log,mean_no_1_h_log)
corr(mean_no_2_m_log,mean_no_2_h_log)

