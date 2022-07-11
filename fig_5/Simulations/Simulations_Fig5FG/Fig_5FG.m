clear;clc;

%% paired looming stimuli; one fast looming
seq_slow = zeros(250, 250, 15);
seq_fast = zeros(250, 250, 15);

L = [-1 : 2/(250) : 1-2/250];
dzs = [0 : 1/250 : 1-1/250];
ixs_slow = [5, 5 : 25 : 150]; 
ixs_fast = [5, 5 : 49 : 250];
[X, Y] = ndgrid(L, L);
Z = X.^2 + Y.^2;

for t = 2 : 8 %% B, L_0, L_0, L_1, ..., 
    seq_slow(:, :, t) = (Z<dzs(1, ixs_slow(1, t-2)));
    seq_fast(:, :, t) = (Z<dzs(1, ixs_fast(1, t-2)));
end

rndNoiseGI = 0;
gi_input_flag = 1;
scale = 10

%%% UL single looming
xsize = 1000; ysize = 800; nT = 15;
As = zeros(xsize, ysize, nT);

As(101:350, 51:300, :) = seq_slow;
str = "single_looming";

helper(As, str, scale, rndNoiseGI, gi_input_flag);

%%% Paired looming
xsize = 1000; ysize = 800; nT = 15;
As = zeros(xsize, ysize, nT);

As(101:350, 51:300, :) = seq_slow;
As(651:900, 501:750, :) = seq_fast;
str = "paired_looming";
helper(As, str, scale, rndNoiseGI, gi_input_flag);

%%% LR looming
xsize = 1000; ysize = 800; nT = 15;
As = zeros(xsize, ysize, nT);

As(651:900, 501:750, :) = seq_fast;
str = "single_strong_looming";
helper(As, str, scale, rndNoiseGI, gi_input_flag);

%%%

ixs.slow = [101:350; 51:300];
ixs.fast = [651:900; 501:750];

data_single = zeros(2, 6, 62500);
data_paired = zeros(2, 6, 62500);

load('./single_looming_data_Wg_scale_10.mat');
for t = 4:8 % L_1, L_2, ...
    img = cd(ixs.slow(1, :), ixs.slow(2, :), t);
    data_single(1, t-2, :) = img(:);
end

load('./single_strong_looming_data_Wg_scale_10.mat');
for t = 4:8 % L_1, L_2, ...,
    img = cd(ixs.fast(1, :), ixs.fast(2, :), t);
end

load('./paired_looming_data_Wg_scale_10.mat');
for t = 4:8 % L_1, L_2, ...,
    img = cd(ixs.slow(1, :), ixs.slow(2, :), t);
    data_paired(1, t - 2, :) = img(:);
end

save('../data_paired_looming.mat', 'data_single', 'data_paired');

