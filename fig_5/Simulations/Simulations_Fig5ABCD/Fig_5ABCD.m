

clear;clc;

%% Static (S)
L = [-1 : 2/250 : 1-2/250];
[X, Y] = ndgrid(L, L);
Z = X .^ 2 + Y .^ 2;
seq_static = zeros(250, 250, 15); % B, B, S, S, S ..., S
for t = 3 : 11
    seq_static(:, :, t) = stim;
end

%%% placing at the centre of 1000 x 800 input space
ixs = [[250+125+1:250+125+250]; [266+1:266+250]];

xsize = 1000; ysize = 800; nT = 15;
As = zeros(xsize, ysize, nT);

As(ixs(1, :), ixs(2, :), :) = seq_static;
str = "static";


%%% simulating static stimuli sequence presentations

helper(As, str);


%% Moving (M)
[Y, X] = ndgrid([-1 : 2/1000 : 1-2/1000], [-1 : 2/1000 : 1-2/1000]);
pos = [0, 0 : 1/50 : 2];

xsize = 1000; ysize = 800; nT = 15;
seq = zeros(xsize, ysize, nT);
for t = 2 : nT % B, S, S, M_1, M_2, ..., M_n
    Z = ((X-pos(1, t-1)).^2 + (Y-pos(1, t-1)).^2) < (0.2480^2);
    seq(:, :, t) = Z(:, 101:900);
end
As = seq;

str = "moving";
helper(As, str);

%%% Looming (L)
L = [-1 : 2/250 : 1-2/250];

dzs = [0 : 1/250 : 1-1/250];
[X, Y] = ndgrid(L, L);

Z = X.^2 + Y.^2;
ixs_l = [5, 5 : 30 : 250];
seq_l = zeros(250, 250, 15);

for t = 2 : 11 % B, L_0, L_0, L_1, ..., L_n
    seq_l(:, :, t) = (Z < dzs(1, ixs_l(1, t - 1)));
end
xsize = 1000; ysize = 800;
nT = 15;

ixs = [[250+125+1:250+125+250]; [266+1:266+250]];
% looming
As = zeros(xsize, ysize, nT);

As(ixs(1, :), ixs(2, :), :) = seq_l;
helper(As, "looming");


%%% Receding (R)
L = [-1 : 2/250 : 1-2/250];
dzs = [0 : 1/250 : 1-1/250];
[X, Y] = ndgrid(L, L);

Z = X.^2 + Y.^2;
ixs_r = [250, 250 : -30 : 5];
seq_r = zeros(250, 250, 15);

for t = 2 : 11 % B, R_0, R_0, R_1, ..., R_n
    seq_r(:, :, t) = (Z < dzs(1, ixs_r(1, t - 1)));
end
xsize = 1000; ysize = 800;
nT = 15;

ixs = [[250+125+1:250+125+250]; [266+1:266+250]];
% looming
As = zeros(xsize, ysize, nT);

As(ixs(1, :), ixs(2, :), :) = seq_r;
helper(As, "receding");

%%%
data = zeros(4, 9, 62500);

strs = {'static', 'moving', 'looming', 'receding'};
ixs = [[250 + 125 + 1 : 250 + 125 + 250]; ...
       [266 + 1 : 266 + 250]];
for i = 1 : 4
    load(sprintf("./%s_data_Wg_scale_1.mat", strs{1, i}));
    for t = 3 : 11
        img = cd(ixs(1, :), ixs(2, :), t);
        data(ix, t-1, :) = img(:);
    end
end
save('../data_static_moving_looming_receding.mat', 'data');
