clear; clc;

load('./data_paired_looming.mat');
strstitle = {'Slow Looming', 'Fast Looming'};
maxV = max(abs([data_single(:); data_paired(:)]));
data_single = data_single./ maxV;
data_paired = data_paired./ maxV;

for ix = 1:2

 subplot(1, 2, ix); hold on;
mu = mean(squeeze(data_single(ix, :, :)), 2);
std_ = std(squeeze(data_single(ix, :, :)), [], 2)./250;

errorbar([2 : 6], mu(2:6, 1), std_(2:6, 1), 'LineWidth', 2, 'Color', 'b');
h = scatter([2 : 6], mu(2:6, 1));
h.Annotation.LegendInformation.IconDisplayStyle = 'off';

mu = mean(squeeze(data_paired(ix, :, :)), 2);
std_ = std(squeeze(data_paired(ix, :, :)), [], 2)./250;

errorbar([2 : 6], mu(2:6, 1), std_(2:6, 1), 'LineWidth', 2, 'Color', 'r');
h = scatter([2 : 6], mu(2:6, 1));
h.Annotation.LegendInformation.IconDisplayStyle = 'off';

if ix == 1
    axis([0, 7, 0.04, 0.06]);
else
    axis([0, 7, 0.08, 0.1]);
end
title(strstitle{1, ix});
end
%% fig 5D (bar plot)
 
%%% taking mean across t = 2 to t = 6 (looming period)
z_singles = mean(data_single, 3); z_singles = mean(z_singles(:, 2:6), 2);
z_paired = mean(data_paired, 3); z_paired = mean(z_paired(:, 2:6), 2);

ratio = 100 * (z_paired - z_singles) ./ z_singles;
bar([1, 2], ratio);
line([0, 3], [0.0, 0.0], 'LineStyle', '--');
axis([0, 3, -8, 8]);
xticklabels({'Slow', 'Fast'});