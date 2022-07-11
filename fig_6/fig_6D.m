clear; clc;
path = "./figures";
A = double(imread('./dataA.png'))./255;
Astar = double(imread('./dataA_star.png'))./255;
CD = double(imread('./CD_image.png'))./255;

ixs_LL = [[451:700]; [151:400]];
ixs_UR = [[101:350]; [651:900]];

Xt = 1- zeros(18, 250, 250, 2);
Yt = 1 - zeros(18, 250, 250, 2);
Yt_cd = 1 - zeros(18, 250, 250, 2);

Xt(1, :, :, 1) = A(ixs_LL(1, :), ixs_LL(2, :));
Xt(1, :, :, 2) = A(ixs_UR(1, :), ixs_UR(2, :));

Xt(16, :, :, 1) = Astar(ixs_LL(1, :), ixs_LL(2, :));
Xt(16, :, :, 2) = Astar(ixs_UR(1, :), ixs_UR(2, :));

for t = 1 : 15
    Yt(t, :, :, 1) = Xt(1, :, :, 1);
    Yt(t, :, :, 2) = Xt(1, :, :, 2);
end
for t = 16 : 18
    Yt(t, :, :, 1) = Xt(16, :, :, 1);
    Yt(t, :, :, 2) = Xt(16, :, :, 2);
end

Yt_cd(1, :, :, 1) = Xt(1, :, :, 1);
Yt_cd(1, :, :, 2) = Xt(1, :, :, 2);

Yt_cd(16, :, :, 1) = CD(ixs_LL(1, :), ixs_LL(2, :));
Yt_cd(16, :, :, 2) = CD(ixs_UR(1, :), ixs_UR(2, :));

% figure; hold on;
% fcount = 1;
% for t = [1 : 18]
%     subplot(3, 18, fcount);
%     img = squeeze(Xt(t, :, :, 1));
%     imshow(img);
%     
%     subplot(3, 18, fcount + 18);
%     img = squeeze(Yt(t, :, :, 1));
%     imshow(img);
% 
%     subplot(3, 18, fcount + 18 * 2);
%     img = squeeze(Yt_cd(t, :, :, 1));
%     imshow(img);
%     fcount = fcount + 1;
% end

%% normal UR vs stimulated UR
figure;
normal_ur = [];
stimulated_ur = [];
for t = 1 : 18

    
    fp = sprintf('%s/GratingsExp_Normal/CD/CD_Normal_%d.png', path, t-1);
    cd_norm_t = double(imread(fp));cd_norm_t = cd_norm_t ./ 255;
    cd_norm_t = cd_norm_t(ixs_UR(1, :), ixs_UR(2, :));
    v1 = cd_norm_t(:); v2 = squeeze(Yt_cd(t, :, :, 2)); v2 = v2(:);
    
    
    measure = mean(abs(v1 - v2));
    normal_ur = [normal_ur; measure];
    
    
    fp = sprintf('%s/GratingsExp_StimUR_BlankPeriod/CD/CD_stimulated_%d.png', path, t-1);
    cd_stim_UR_t = double(imread(fp));cd_stim_UR_t = cd_stim_UR_t ./ 255;
    cd_stim_UR_t = cd_stim_UR_t(ixs_UR(1, :), ixs_UR(2, :));
    
    v1 = cd_stim_UR_t(:); v2 = squeeze(Yt_cd(t, :, :, 2)); v2 = v2(:);
    measure = mean(abs(v1 - v2));
    
    stimulated_ur = [stimulated_ur; measure];
end

subplot(1, 2, 1); hold on;
plot(normal_ur(1:18, 1), 'LineWidth', 2, 'Color', 'r');
scatter([1 : 3 : 18], normal_ur(1: 3 :18, 1));

plot(stimulated_ur(1 : 18, 1), 'LineWidth', 2, 'Color', 'b');
scatter([1: 3 : 18], stimulated_ur(1: 3 : 18, 1));
xlabel("Time"); ylabel("L1 error");
legend('Without \mu - stim.', 'With \mu - stim.');

axis([0, 19, 0, 0.3]);
title("\mu - stim at UR");
yticks([0, 0.3]);
%%%
normal_ll = [];
stimulated_ll = [];

for t = 1 : 18

    fp = sprintf('%s/GratingsExp_Normal/CD/CD_Normal_%d.png', path, t-1);
    cd_norm_t = double(imread(fp));cd_norm_t = cd_norm_t ./ 255;
    cd_norm_t = cd_norm_t(ixs_LL(1, :), ixs_LL(2, :));
    
    v1 = cd_norm_t(:); v2 = squeeze(Yt_cd(t, :, :, 1)); v2 = v2(:);
    measure = mean(abs(v1 - v2));
    
    normal_ll = [normal_ll; measure];
    
    fp = sprintf('%s/GratingsExp_StimLL_BlankPeriod/CD/CD_stimulated_%d.png', path, t-1);
    cd_stim_LL_t = double(imread(fp)); cd_stim_LL_t = cd_stim_LL_t ./ 255;
    cd_stim_LL_t = cd_stim_LL_t(ixs_LL(1, :), ixs_LL(2, :));
    
    v1 = cd_stim_LL_t(:); v2 = squeeze(Yt_cd(t, :, :, 1)); v2 = v2(:);
    
    measure = mean(abs(v1 - v2));
    stimulated_ll = [stimulated_ll; measure];
end

subplot(1, 2, 2);
hold on;
plot(normal_ll, 'LineWidth', 2, 'Color', 'r');
scatter([1:3:18], normal_ll(1:3:18, 1));

plot(stimulated_ll, 'LineWidth', 2, 'Color', 'b');
scatter([1:3 : 18], stimulated_ll(1:3:18, 1));
%plot([2 : 18], stimulated_ll(2 : 18, 1), 'LineWidth', 2, 'Color', 'b');

legend("Without \mu - stim.", "With \mu - stim.");
xlabel("Time"); ylabel("L1 error");
title("\mu - stim at LL");
yticks([0, 0.3]);
axis([0, 19, 0, 0.3]);
