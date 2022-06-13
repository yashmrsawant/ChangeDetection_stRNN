
clear; clc;
load('./data_static_moving_looming_receding.mat');
strstitle = {'Static', 'Moving', 'Looming', 'Receding'};
maxV = max(abs(data(:)));
data = data./ maxV;
for ix = 1:4
subplot(1, 4, ix); hold on;
mu = mean(squeeze(data(ix, :, :)), 2);
std_ = std(squeeze(data(ix, :, :)), [], 2)./250;


    h = scatter([1:2], mu(1:2, 1));
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    errorbar([3 : 10], mu(3:10, 1), std_(3:10, 1), 'LineWidth', 2);
    h = scatter([3:10], mu(3:10, 1));
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';

    axis([0, 10, 0, 0.1]);

title(strstitle{1, ix});
end

%% fig 5B
figure;
Z = mean(data(:, 3 : end, :), 3);
mu_Z = mean(Z, 2);
bar(mean(Z, 2));
xticks([1:4]);
xticklabels({'Static', 'Moving', 'Looming', 'Receding'});


