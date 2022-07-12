clear; clc;

Wx = dlmread('./Wx2237_per.csv');
Wo = dlmread('./Wo2237_per.csv');
Wh = dlmread('./Wh2237_per.csv');

%% plotting E neurons kernel
load('./none_silenced.mat');
hidden_s = hidden_activity; hidden_s(hidden_s < 0) = 0;
mu_act_mean = squeeze(mean(mean(hidden_s(:, 6:end, :), 2), 1));
var_act_mean = squeeze(mean(var(hidden_s(:, 6:end, :), [], 2), 1));

ignore_units = find(mu_act_mean < 0.01);
[~, sort_ixs] = sort(var_act_mean);

stability_order = []; nN = size(sort_ixs, 1);
for i = 1 : nN
    if isempty(find(ignore_units == sort_ixs(i, 1)))
        stability_order = [stability_order, sort_ixs(i, 1)];
    end
end
top10_E.stable = [];
top10_E.unstable = [];

for i = 1 : length(stability_order)
    if stability_order(1, i) < 256
        top10_E.stable = [top10_E.stable, stability_order(1, i)];
    end
end
for i = length(stability_order) : -1 : 1
    if stability_order(1, i) < 256
        top10_E.unstable = [top10_E.unstable, stability_order(1, i)];
    end
end


stable_ixs = [1 : 10];
unstable_ixs = [1 : 10];
%%
colormap_ = [[0:0.01:1]', [0:0.01:1]', [0:0.01:1]'];

%% stable E neurons

%%% unit_pos (16 x 16), input (8 x 8), E-E (In, 16 x 16), E-E(Out, 16 x
%%% 16), output (8 x 8)

h = figure;
rcount = 1;
for i = 1:10
ix = top10_E.stable(1, i);

img = zeros(256, 1); img(ix, 1) = 1;
img = reshape(img, 16, 16);

subplot(10, 5, rcount);
imshow(1-img);
axis square;
axis off;
sz_p = size(img, 1); sz_q = size(img, 2);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
caxis([0, 1]);
colorbar;
colormap(colormap_);

subplot(10, 5, 1+rcount);
img = reshape(Wx(:, ix), 8, 8);
imagesc(1-img);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
caxis([0, 1]);
colorbar;
colormap(colormap_);
axis square;
axis off;

subplot(10, 5, 2+rcount);
img = reshape(Wh(1:256, ix), 16, 16);
imagesc(1-img);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
caxis([0, 1]);
colorbar;
colormap(colormap_);
axis square;
axis off;

subplot(10, 5, 3+rcount);
img = reshape(Wh(ix, 1:256), 16, 16);
imagesc(1-img);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
caxis([0, 1]);
colorbar;
colormap(colormap_);
axis square;
axis off;

subplot(10, 5, 4+rcount);
img = reshape(Wo(ix, :), 8, 8);
imagesc(1-img);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
caxis([0, 1]);
colorbar;
colormap(colormap_);
axis square;
axis off;

rcount = rcount + 5;
end
sgtitle("Stable E neurons kernels");
saveas(h, "./stable.svg");
close(h);

%% unstable E neurons

%%% unit_pos (16 x 16), input (8 x 8), E-E (In, 16 x 16), E-E(Out, 16 x
%%% 16), output (8 x 8)

h = figure;
rcount = 1;
for i = 1:10
ix = top10_E.unstable(1, i);

img = zeros(256, 1); img(ix, 1) = 1;
img = reshape(img, 16, 16);

subplot(10, 5, rcount);
imshow(1-img);
axis square;
axis off;
sz_p = size(img, 1); sz_q = size(img, 2);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
caxis([0, 1]);
colorbar;
colormap(colormap_);


subplot(10, 5, 1+rcount);
img = reshape(Wx(:, ix), 8, 8);
imagesc(1-img);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
caxis([0, 1]);
colorbar;
colormap(colormap_);
axis square;
axis off;

subplot(10, 5, 2+rcount);
img = reshape(Wh(1:256, ix), 16, 16);
imagesc(1-img);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
caxis([0, 1]);
colorbar;
colormap(colormap_);
axis square;
axis off;

subplot(10, 5, 3+rcount);
img = reshape(Wh(ix, 1:256), 16, 16);
imagesc(1-img);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
caxis([0, 1]);
colorbar;
colormap(colormap_);
axis square;
axis off;

subplot(10, 5, 4+rcount);
img = reshape(Wo(ix, :), 8, 8);
imagesc(1-img);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
rectangle('position', [0.5, 0.5, sz_q, sz_p], 'edgecolor', [0, 0, 0]);
caxis([0, 1]);
colorbar;
colormap(colormap_);
axis square;
axis off;

rcount = rcount + 5;
end
sgtitle("Unstable E neurons kernels");
saveas(h, "./unstable.svg");
close(h);
