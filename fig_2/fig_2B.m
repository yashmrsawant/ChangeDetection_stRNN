clear; clc;
load('./Fig_2B/none_silenced.mat');
load('./Fig_2B/top_16_stable_silenced.mat');
load('./Fig_2B/top_16_unstable_silenced.mat');
%%
t2_meanInput = squeeze(sum(input_fed(:, 2, :), 3));
patterns = [9, 237, 428, 240, 246, 299, 75, 245, 206, 5, 298, 180] + 1;
%%
colors = [
[0 0.4470 0.7410];
[0.8500 0.3250 0.0980];
[0.9290 0.6940 0.1250];
[0.4940 0.1840 0.5560];
[0.4660 0.6740 0.1880];
[0.3010 0.7450 0.9330];
[0.6350 0.0780 0.1840]];

%% activity - stable and unstable units
colors = min([colors; colors([2, 3, 5, 1, 4, 6, 7], :) + 0.01 * rand(1, 3); ...
                colors([2, 1], :) + 0.01 * rand(1, 3)], 1);
idxs = [1 : 16];
for i = 1 : 2
   h = figure('Position', [0, 0, 500, 800]);
   subplot(2, 1, 1); hold on;
   y = reshape(max(hidden_activity(patterns(i), 1 : 20, stable_ixs + 1), 0), 20, 16);
   for idx = idxs
       plot([1 : 3 : 20], y(1 : 3 : 20, idx), 'Color', colors(idx, :));
       scatter([1:3:20], y(1 : 3 : 20, idx), 70, 'MarkerEdgeColor', colors(idx, :));
   end
   axis([0, 20, 0.0, 1]);
%    axes('Position', [.39, .94, .05, .05]);
%    box on
%    imshow(1 - reshape(target(patterns(i), 2, :), 8, 8));
%    axis off;

   subplot(2, 1, 2); hold on;
   y = reshape(max(hidden_activity(patterns(i), 1 : 20, unstable_ixs + 1), 0), 20, 16);
   for idx = idxs
       plot([1 : 3 : 20], y(1 : 3 : 20, idx), 'Color', colors(idx, :));
       scatter([1:3:20], y(1 : 3 : 20, idx), 70, 'MarkerEdgeColor', colors(idx, :));
   end
   axis([0, 20, 0, 1]);   
   sgtitle(strcat("     sample #", num2str(i)));

   saveas(h, strcat('./Fig_2B/fig_n', num2str(i), '.jpg'));
end
%% stimulus input pattern - i = 3 to 12 (extended data figure 2B)
for i = 1 : 2
    h = figure;
    tg = imresize(1 - reshape(target(patterns(i), 2, :), 8, 8), ...
        30, 'nearest');
    imshow(tg);  
    saveas(h, strcat('./Fig_2C/fig_pattern_', num2str(i), '.jpg'));
    close;
end