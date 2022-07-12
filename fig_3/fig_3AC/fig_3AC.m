clear; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fig 3B %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nunits = 16;
seq_l = 20;
load('./none_silenced.mat');
stable = zeros(500, seq_l);
filename = sprintf("./top_%d_stable_silenced.mat", ...
    nunits);
load(filename);
counter = 1
for i = 1 : 500
    for j = 1 : seq_l
        tg = reshape(target(i, j, :), 64, 1);
        pr = reshape(prediction(i, j, :), 64, 1);
        stable(counter, j) = sum(abs(tg - pr) / 64);
    end
    counter = counter + 1;
end

Stable_Sile_error = stable;
% plot(squeeze(mean(Stable_Sile_error, 1))', 'color', 'r', 'LineWidth', 2);

% unstable
unstable = zeros(500, seq_l);
load('./none_silenced.mat');
filename = sprintf("./top_%d_unstable_silenced", nunits);
load(filename);
counter = 1;
for i = 1 : 500
    for j = 1 : seq_l
        tg = reshape(target(i, j, :), 64, 1);
        pr = reshape(prediction(i, j, :), 64, 1);
        unstable(counter, j) = sum(abs(tg - pr) / 64);
    end
    counter = counter + 1;
end


UnStable_Sile_error = unstable;
% plot(squeeze(mean(UnStable_Sile_error, 1)), 'color', 'b', 'LineWidth', 2);

% none

filename = sprintf("./none_silenced.mat");
load(filename);

seq_l = 20;
normal = zeros(500, seq_l);

counter = 1;
for i = 1 : 500
    for j = 1 : seq_l
        tg = reshape(target(i, j, :), 64, 1);
        pr = reshape(prediction(i, j, :), 64, 1);
        normal(counter, j) = sum(abs(tg - pr) / 64);
    end
    counter = counter + 1;
end
% plot(squeeze(mean(normal, 1)), 'color', 'k', 'LineWidth', 1);
% legend('Stable', 'Unstable', 'None');
% axis([0, 20, 0, 0.3]);

%%% Plotting
fromId = 3;
toId = 20;
normalm = normal(:, fromId : toId);
normalm(sum(isnan(normalm), 2) == 1, :) = [];

normalmax = max(normalm, [], 1);
normalmin = min(normalm, [], 1);

stablem = stable(:, fromId : toId);
stablem(sum(isnan(stablem), 2) == 1, :) = [];

stablemax = max(stablem, [], 1);
stablemin = min(stablem, [], 1);

unstablem = unstable(:, fromId : toId);
unstablem(sum(isnan(unstablem), 2) == 1, :) = [];
unstablemax = max(unstablem, [], 1);
unstablemin = min(unstablem, [], 1);

figure; hold on;

% plot([1 : sseq], mean(normalm), 'LineWidth', 2); hold on;
shadedErrorBar([fromId : toId], mean(normalm, 1), std(normalm) / sqrt(250), 'lineprops', 'g');
Z = mean(normalm, 1);
h = scatter([fromId : 2 : toId], Z(1, 1 : 2 : end), 10);
h.Annotation.LegendInformation.IconDisplayStyle = 'off';

% plot([1 : sseq], mean(stablem), 'LineWidth', 2); 
shadedErrorBar([fromId : toId], mean(stablem, 1), std(stablem) / sqrt(250), 'lineprops', 'r');
Z = mean(stablem, 1);
h = scatter([fromId : 2 : toId], Z(1, 1 : 2 : end), 10);
h.Annotation.LegendInformation.IconDisplayStyle = 'off';

% plot([1 : sseq], mean(unstablem), 'LineWidth', 2); 
shadedErrorBar([fromId : toId], mean(unstablem, 1), std(unstablem) / sqrt(250), 'lineprops', 'b');
Z = mean(unstablem, 1);
h = scatter([fromId : 2 : toId], Z(1, 1 : 2 : end), 10);
h.Annotation.LegendInformation.IconDisplayStyle = 'off';

axis([2, 20, 0, 0.3]);
yticks([0.1, 0.2, 0.3, 0.4, 0.5]);
legend('None', 'Stable', 'Unstable');
xlabel('Time');
ylabel('Reconstruction Error');
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fig 3A%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
load("./none_silenced.mat");
load('./top_16_stable_silenced.mat');
load('./top_16_unstable_silenced.mat');

hidden_s = max(hidden_activity, 0);
mean_act_var = mean(squeeze(var(hidden_s(:, 6 : end, :), [], 2)), 1);
sorted_mean_act_var = sort(mean_act_var);


stable_act_var = mean_act_var(1, stable_ixs + 1);
mu_st_act_var = mean(stable_act_var);
std_st_act_var = std(stable_act_var);
fprintf("Stable: mean = (%d +/- %d)x %f\n", mu_st_act_var * 1e5, ...
    ceil(std_st_act_var * 1e5), 1e-5);

max_stable_act_var = max(stable_act_var);


unstable_act_var = mean_act_var(1, unstable_ixs + 1);
mu_unst_act_var = mean(unstable_act_var);
std_unst_act_var = std(unstable_act_var);
fprintf("Unstable: mean = (%d +/- %d)x %f\n", mu_unst_act_var * 1e5, ...
    std_unst_act_var * 1e5, 1e-5);
min_unstable_act_var = min(unstable_act_var);


fprintf("Unstable/Stable: mean = %f\n", mu_unst_act_var/mu_st_act_var);
figure; hold on;
histogram(mean_act_var);
scatter(mu_unst_act_var, 16, 'v');
errorbar(mu_unst_act_var, 16, std_unst_act_var, 'horizontal');
line([max_stable_act_var, max_stable_act_var], [0, 30], 'Color', 'k');
line([min_unstable_act_var, min_unstable_act_var], [0, 30], 'Color', 'k', 'LineWidth', 2);
axis([0 - 0.00001, max(mean_act_var) + 0.01, 0, 30]);