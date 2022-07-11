function helper(As, str, scale, rndNoiseGI, gi_input_flag)
% change detection network
%%% cd-stRNN

path = '../weights_cd/Epoch_7280_iter_0.h5'
info = h5info(path);
Wh_cd = double(h5read(path, '/Wh'))';
Wo_cd = double(h5read(path, '/Wo'))';
Wx_cd = double(h5read(path, '/Wx'))';
bo_cd = double(h5read(path, '/bo'))';

st_cd = @(xt, rtm1) tanh(xt * Wx_cd + rtm1 * Wh_cd);
rt_cd = @(st_v) st_v .* double(st_v > 0);
ot_cd = @(rt)  1./(1+exp(-1*(rt * Wo_cd + bo_cd)));
%%% mc-stRNN
epoch_num = 100;
%%% mc-stRNN with GI having excitatory input from stRNN hidden unit with
%%% recurrent I-I connections
info = h5info(sprintf('../weights_mc/weights_epoch%d.h5', epoch_num));

Wg_x = double(h5read(sprintf('../Weights_mc/weights_epoch%d.h5', epoch_num), '/Wg_x')');
Wg_s = double(h5read(sprintf('../Weights_mc/weights_epoch%d.h5', epoch_num), '/Wg_s')');
Wg_g = double(h5read(sprintf('../Weights_mc/weights_epoch%d.h5', epoch_num), '/Wg_g')');
Wg = double(h5read(sprintf('../Weights_mc/weights_epoch%d.h5', epoch_num), '/Wg')');
Wx = double(h5read(sprintf('../Weights_mc/weights_epoch%d.h5', epoch_num), '/Wx')');
Wh = double(h5read(sprintf('../Weights_mc/weights_epoch%d.h5', epoch_num), '/Wh')');
Wo = double(h5read(sprintf('../Weights_mc/weights_epoch%d.h5', epoch_num), '/Wo')');
bo = double(h5read(sprintf('../Weights_mc/weights_epoch%d.h5', epoch_num), '/bo')');

gt_s = @(xg, gtm1, rtm1) tanh(xg * Wg_x .* gi_input_flag + gtm1 * Wg_g + rtm1(1, 1:256) * Wg_s);
gt = @(gt_v) gt_v .* (gt_v > 0);
rt_mc = @(st_v) st_v .* (st_v > 0);
ot_mc = @(rt) 1./(1+exp(-1 * (rt * Wo + bo)));
st_mc = @(xt, gt, rtm1) tanh(xt * Wx + rtm1 * Wh + gt * Wg .* scale);
nX = size(Wx, 1); nG = size(Wg, 1); nN = size(Wh, 1); nT = size(As, 3);

per = zeros(size(As));
cd = zeros(size(As));

rtm1s = rand(12500, nN, 4) * 0.0;
gtm1 = zeros(1, nG);
for t = 1 : nT
    for k = 1 : 4
        Ots = zeros(12500, nX);

        frame = As(:, :, t);
        xg = min(imresize(frame, [10, 10], 'nearest') + rndNoiseGI, 1);
        M = partitionize(frame, k);
        gtm1 = gt(gt_s(xg(:)', gtm1, mean(rtm1s(:, :, k), 1) .* 1));
        for i = 1 : 12500
            rtm1s(i, :, k) = rt_mc(st_mc(M(i, :), gtm1, rtm1s(i, :, k)));
            Ots(i, :) = ot_mc(rtm1s(i, :, k));
        end
        part = departitionize(Ots, k);
        per(5:1000-4, 5:800-4, t) = per(5:1000-4, 5:800-4, t) + ...
                                    part(5:1000-4, 5:800-4);
    end
    fprintf("frame %d processed\n", t);    
end

% cd
rtm1s = zeros(12500, nN, 4);
for t = 1 : nT
    for k = 1 : 4
        Ots = zeros(12500, nX);
        frame = per(:, :, t)./4;
        M = partitionize(frame, k);            

        for i = 1 : 12500
            rtm1s(i, :, k) = rt_cd(st_cd(M(i, :), rtm1s(i, :, k)));
            Ots(i, :) = ot_cd(rtm1s(i, :, k));
        end
        part = departitionize(Ots, k);
        cd(5:1000-4, 5:800-4, t) = cd(5:1000-4, 5:800-4, t) + part(5:1000-4, 5:800-4);
    end
    fprintf("frame %d processed\n", t);    
end
if ~exist(sprintf('./Figures/%s/', str), 'dir')
    mkdir(sprintf('./Figures/%s/', str));
end

if ~exist(sprintf('./Figures/%s/per_%d/', str, scale), 'dir')
    mkdir(sprintf('./Figures/%s/per_%d/', str, scale), 'dir');
end
if ~exist(sprintf('./Figures/%s/cd_%d/', str, scale), 'dir')
    mkdir(sprintf('./Figures/%s/cd_%d/', str, scale), 'dir');
end
if ~exist(sprintf('./Figures/%s/in/', str), 'dir')
    mkdir(sprintf('./Figures/%s/in/', str), 'dir');
end
if ~exist(sprintf('./Figures/%s/exp_cd/', str), 'dir')
    mkdir(sprintf('./Figures/%s/exp_cd/', str), 'dir');
end


for t = 1 : nT
    img = As(:, :, t);
    h = figure('Position', get(0, 'Screensize'), 'Visible', 'off');
    imshow(1-img');
    saveas(h, sprintf("./Figures/%s/in/inp_%d.jpg", str, t));
    close(h);
        
    if t > 1
        img = abs(As(:, :, t) - As(:, :, t-1));
        h = figure('Position', get(0, 'Screensize'), 'Visible', 'off');
        imshow(1-img');
        saveas(h, sprintf("./Figures/%s/exp_cd/expCD_%d.jpg", str, t));
        close(h);
    end
    img = per(:, :, t)./4;
    h = figure('Position', get(0, 'Screensize'), 'Visible', 'off');
    imshow(1-img');
    
    saveas(h, sprintf("./Figures/%s/per_%d/per_%d.jpg", str, scale, t));
    close(h);
    
    img = cd(:, :, t)./4;
    h = figure('Position', get(0, 'Screensize'), 'Visible', 'off');
    imshow(1-img');
    saveas(h, sprintf("./Figures/%s/cd_%d/cd_%d.jpg", str, scale, t));
    close(h);
end
save(sprintf("./%s_data_Wg_scale_%d.mat", str, scale), 'As', 'per', 'cd');
end
