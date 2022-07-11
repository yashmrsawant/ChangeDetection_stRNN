close all; clear; clc;
%% 8D
load('./sal_metrics_for_visualization_akshay.mat');
metrics = sal_metrics_for_visualization;

metric_labels = {'AUC Judd', 'AUC Borji', 'KL Div', 'SIM', 'CC'};

% metric_id = 2;
% xs = metrics(8 : end, ((metric_id - 1) * 3) + 1);
% ys_sali = metrics(8 : end, ((metric_id - 1) * 3) + 2); % salicon
% ys_itti = metrics(8 : end, ((metric_id - 1) * 3) + 3); % itti

figure;
% salicon vs model
subplot(2, 2, 1);
metric_id = 2; metric_labels{1, metric_id}
xs = metrics(8 : end, ((metric_id - 1) * 3) + 1);
ys_sali = metrics(8 : end, ((metric_id - 1) * 3) + 2); % salicon


scatter(xs, ys_sali); hold on;
plot(0 : 1/ 100 : 1, 0 : 1/100 : 1, '--k');
xlim([0.45, 0.75]); ylim([0.45, 0.75]);
xlabel('stRNN');
ylabel('Salicon');
title(strcat(metric_labels{1, metric_id}, ', p-value:', num2str(signrank(xs, ys_sali))));
axis square;
axes('Position', [.2, .8, .1, .1]);
box on
dif = xs - ys_sali;
histogram(dif, linspace(-max(abs(dif)), max(abs(dif)), 20));
axis square;
line([0, 0], [0, 10], 'LineStyle', '--', 'Color', 'k');
subplot(2, 2, 2);
metric_id = 3; metric_labels{1, metric_id}
xs = metrics(8 : end, ((metric_id - 1) * 3) + 1);
ys_sali = metrics(8 : end, ((metric_id - 1) * 3) + 2); % salicon
scatter(xs, ys_sali); hold on;
plot(4.5 : 1/ 100 : 10.2, 4.5 : 1/100 : 10.2, '--k');
xlim([4.5, 10.2]); ylim([4.5, 10.2]);
xlabel('stRNN');
ylabel('Salicon');
title(strcat(metric_labels{1, metric_id}, ', p-value:', num2str(signrank(xs, ys_sali))));
axis square;
axes('Position', [.73, .6, .1, .1]);
box on
dif = xs - ys_sali;
histogram(dif, linspace(-max(abs(dif)), max(abs(dif)), 20));
axis square;
line([0, 0], [0, 5], 'LineStyle', '--', 'Color', 'k');

subplot(2, 2, 3);
metric_id = 4; metric_labels{1, metric_id}
xs = metrics(8 : end, ((metric_id - 1) * 3) + 1);
ys_sali = metrics(8 : end, ((metric_id - 1) * 3) + 2); % salicon

scatter(xs, ys_sali); hold on;
plot(0 : 1/ 100 : 1, 0 : 1/100 : 1, '--k');
xlim([0.35, 0.85]); ylim([0.35, 0.85]);
xlabel('stRNN');
ylabel('Salicon');
title(strcat(metric_labels{1, metric_id}, ', p-value:', num2str(signrank(xs, ys_sali))));
axis square;
axes('Position', [.2, .3, .1, .1]);
box on
dif = xs - ys_sali;
histogram(dif, linspace(-max(abs(dif)), max(abs(dif)), 20));
axis square;
line([0, 0], [0, 8], 'LineStyle', '--', 'Color', 'k');

subplot(2, 2, 4);
metric_id = 5; metric_labels{1, metric_id}
xs = metrics(8 : end, ((metric_id - 1) * 3) + 1);
ys_sali = metrics(8 : end, ((metric_id - 1) * 3) + 2); % salicon

scatter(xs, ys_sali); hold on;
plot(-0.12 : 1/ 100 : 1, -0.12 : 1/100 : 1, '--k');
xlim([-0.12, 1.2]); ylim([-0.12, 1.2]);
xlabel('stRNN');
ylabel('Salicon');
title(strcat(metric_labels{1, metric_id}, ', p-value:', num2str(signrank(xs, ys_sali))));
axis square;
axes('Position', [.64, .33, .1, .1]);
box on
dif = xs - ys_sali;
histogram(dif, linspace(-max(abs(dif)), max(abs(dif)), 20));
axis square;
line([0, 0], [0, 4], 'LineStyle', '--', 'Color', 'k');

%% 8E
figure;
% itti vs model
subplot(2, 2, 1);
metric_id = 2; metric_labels{1, metric_id}
xs = metrics(8 : end, ((metric_id - 1) * 3) + 1);
ys_itti = metrics(8 : end, ((metric_id - 1) * 3) + 3); % itti

scatter(xs, ys_itti); hold on;
plot(0 : 1/ 100 : 1, 0 : 1/100 : 1, '--k');
xlim([0.45, 0.75]); ylim([0.45, 0.75]);
xlabel('stRNN');
ylabel('Itti-Koch');
title(strcat(metric_labels{1, metric_id}, ', p-value:', num2str(signrank(xs, ys_itti))));
axis square;
axes('Position', [.2, .8, .1, .1]);
box on
dif = xs - ys_itti;
histogram(dif, linspace(-max(abs(dif)), max(abs(dif)), 20));
axis square;
line([0, 0], [0, 10], 'LineStyle', '--', 'Color', 'k');
subplot(2, 2, 2);
metric_id = 3; metric_labels{1, metric_id}
xs = metrics(8 : end, ((metric_id - 1) * 3) + 1);
ys_itti = metrics(8 : end, ((metric_id - 1) * 3) + 3); % itti
scatter(xs, ys_itti); hold on;

plot(4.5 : 1/ 100 : 7.2, 4.5 : 1/100 : 7.2, '--k');
xlim([4.5, 7.2]); ylim([4.5, 7.2]);
xlabel('stRNN');
ylabel('Itti-Koch');
title(strcat(metric_labels{1, metric_id}, ', p-value:', num2str(signrank(xs, ys_itti))));
axis square;
axes('Position', [.73, .6, .1, .1]);
box on
dif = xs - ys_itti;
histogram(dif, linspace(-max(abs(dif)), max(abs(dif)), 20));
axis square;
line([0, 0], [0, 10], 'LineStyle', '--', 'Color', 'k');

subplot(2, 2, 3);
metric_id = 4; metric_labels{1, metric_id}
xs = metrics(8 : end, ((metric_id - 1) * 3) + 1);
ys_itti = metrics(8 : end, ((metric_id - 1) * 3) + 3); % itti
scatter(xs, ys_itti); hold on;
plot(0 : 1/ 100 : 1, 0 : 1/100 : 1, '--k');
xlim([0.43, 0.85]); ylim([0.43, 0.85]);
xlabel('stRNN');
ylabel('Itti-Koch');
title(strcat(metric_labels{1, metric_id}, ', p-value:', num2str(signrank(xs, ys_itti))));
axis square;
axes('Position', [.2, .3, .1, .1]);
box on
dif = xs - ys_itti;
histogram(dif, linspace(-max(abs(dif)), max(abs(dif)), 20));
axis square;
line([0, 0], [0, 10], 'LineStyle', '--', 'Color', 'k');

subplot(2, 2, 4);
metric_id = 5; metric_labels{1, metric_id}
xs = metrics(8 : end, ((metric_id - 1) * 3) + 1);
ys_itti = metrics(8 : end, ((metric_id - 1) * 3) + 3); % itti
scatter(xs, ys_itti); hold on;
plot(-0.12 : 1/ 100 : 1, -0.12 : 1/100 : 1, '--k');
xlim([-0.12, 0.9]); ylim([-0.12, 0.9]);
xlabel('stRNN');
ylabel('Itti-koch');
title(strcat(metric_labels{1, metric_id}, ', p-value:', num2str(signrank(xs, ys_itti))));
axis square;
axes('Position', [.64, .33, .1, .1]);
box on
dif = xs - ys_itti;
histogram(dif, linspace(-max(abs(dif)), max(abs(dif)), 20));
axis square;
line([0, 0], [0, 10], 'LineStyle', '--', 'Color', 'k');