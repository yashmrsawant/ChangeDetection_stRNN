clear all; clc;
%%
Wh = dlmread('./Weights/Wh2237_per.csv');
Wo = dlmread('./Weights/Wo2237_per.csv');
Wx = dlmread('./Weights/Wx2237_per.csv');
bo = dlmread('./Weights/bo2237_per.csv');
bo = bo';
bi = dlmread('./Weights/bi2237_per.csv');
nN = size(Wh, 1); nX = size(Wx, 1);


%% stimulus sequence
nframes = 30;
Xt = zeros(nframes, nX);
Yt = zeros(nframes, nX);
pX = 0.2;
st_xt = rand(1, nX) < 0.5;
Xt(1, :) = st_xt;
Xg(1, :) = 1;
for i = 1 : (nframes / 2)
    Yt(i, :) = Xt(1, :);
end
st_xt = rand(1, nX) < 0.5;
Xt((nframes / 2 + 1), :) = st_xt;
for i = (nframes / 2 + 1 : 25)
    Yt(i, :) = st_xt;
end
Xt(26, :) = rand(1, nX) < 0.5;
for i = 26 : nframes
    Yt(i, :) = Xt(26, :);
end
%% network
st = @(xt, rtm1) tanh(xt * Wx + rtm1 * Wh);
rt = @(st_v) st_v .* double(st_v > 0);
ot = @(rt) 1 ./ (1 + exp(-1 * (rt * Wo + bo)));
%% running network
rtm1 = zeros(1, nN); 
Ot = zeros(nframes, nX);
h = zeros(nframes, nN);
for i = 1 : nframes
    rtm1 = rt(st(Xt(i, :), rtm1));
    h(i, :) = rtm1;
    Ot(i, :) = ot(rtm1);
end

%% %%%========= Deactivating flags EE|II|EI|IE =============%%%
deactivate_flag_ee = ones(nN, nN); deactivate_flag_ee(1 : 256, 1 : 256) = 0;
deactivate_flag_ii = ones(nN, nN); deactivate_flag_ii(257 : end, 257 : end) = 0;
deactivate_flag_ie = ones(nN, nN); deactivate_flag_ie(257 : end, 1 : 256) = 0;
deactivate_flag_ei = ones(nN, nN); deactivate_flag_ei(1 : 256, 257 : end) = 0;
%% EE
Wh_s = Wh .* deactivate_flag_ee;
imshow(Wh_s ~= 0);
st_s = @(xt, rtm1) tanh(xt * Wx + rtm1 * Wh_s);

%%==== silencing at start === %%
scale = 0.01;
rtm1 = rand(1, nN) * scale; 
Ot = zeros(nframes, nX);
rtm1 = rt(st(Xt(1, :), rtm1));
Ot(1, :) = ot(rtm1);

for i = 1 : ((nframes / 2))
    rtm1 = rt(st(Xt(i, :), rtm1));
    Ot(i, :) = ot(rtm1);
end
i = nframes / 2 + 1;
rtm1 = rt(st_s(Xt(i, :), rtm1));
Ot(i, :) = ot(rtm1);
 
for i = [(nframes / 2 + 2) : nframes]
    rtm1 = rt(st(Xt(i, :), rtm1));
    Ot(i, :) = ot(rtm1);
end
 
plot_result(Xt(nframes/2 - 1 : end, :), Yt(nframes/2 - 1 : end, :), ...
    Ot(nframes/2 - 1 : end, :), [3], [0, 1, 0]);
%% II
Wh_s = Wh .* deactivate_flag_ii;
imshow(Wh_s ~= 0);
st_s = @(xt, rtm1) tanh(xt * Wx + rtm1 * Wh_s);

%%==== silencing at start === %%
scale = 0.01;
rtm1 = rand(1, nN) * scale; 
Ot = zeros(nframes, nX);
rtm1 = rt(st(Xt(1, :), rtm1));
Ot(1, :) = ot(rtm1);

for i = 1 : ((nframes / 2))
    rtm1 = rt(st(Xt(i, :), rtm1));
    Ot(i, :) = ot(rtm1);
end
i = nframes / 2 + 1;
rtm1 = rt(st_s(Xt(i, :), rtm1));
Ot(i, :) = ot(rtm1);
 
for i = [(nframes / 2 + 2) : nframes]
    rtm1 = rt(st(Xt(i, :), rtm1));
    Ot(i, :) = ot(rtm1);
end
 
plot_result(Xt(nframes/2 - 1 : end, :), Yt(nframes/2 - 1 : end, :), ...
    Ot(nframes/2 - 1 : end, :), [3], [0, 0, 0])

%% IE
Wh_s = Wh .* deactivate_flag_ie;
imshow(Wh_s ~= 0);
st_s = @(xt, rtm1) tanh(xt * Wx + rtm1 * Wh_s);

%%==== silencing at start === %%
scale = 0.01;
rtm1 = rand(1, nN) * scale; 
Ot = zeros(nframes, nX);
rtm1 = rt(st(Xt(1, :), rtm1));
Ot(1, :) = ot(rtm1);

for i = 1 : ((nframes / 2))
    rtm1 = rt(st(Xt(i, :), rtm1));
    Ot(i, :) = ot(rtm1);
end
i = nframes / 2 + 1;
rtm1 = rt(st_s(Xt(i, :), rtm1));
Ot(i, :) = ot(rtm1);
 
for i = [(nframes / 2 + 2) : nframes]
    rtm1 = rt(st(Xt(i, :), rtm1));
    Ot(i, :) = ot(rtm1);
end
 
plot_result(Xt(nframes/2 - 1 : end, :), Yt(nframes/2 - 1 : end, :), ...
    Ot(nframes/2 - 1 : end, :), [3], [0, 0, 1])

%% EI
Wh_s = Wh .* deactivate_flag_ei;
imshow(Wh_s ~= 0);
st_s = @(xt, rtm1) tanh(xt * Wx + rtm1 * Wh_s);

%%==== silencing at start === %%
scale = 0.01;
rtm1 = rand(1, nN) * scale; 
Ot = zeros(nframes, nX);
rtm1 = rt(st(Xt(1, :), rtm1));
Ot(1, :) = ot(rtm1);

for i = 1 : ((nframes / 2))
    rtm1 = rt(st(Xt(i, :), rtm1));
    Ot(i, :) = ot(rtm1);
end
i = nframes / 2 + 1;
rtm1 = rt(st_s(Xt(i, :), rtm1));
Ot(i, :) = ot(rtm1);
 
for i = [(nframes / 2 + 2) : nframes]
    rtm1 = rt(st(Xt(i, :), rtm1));
    Ot(i, :) = ot(rtm1);
end
 
plot_result(Xt(nframes/2 - 1 : end, :), Yt(nframes/2 - 1 : end, :), ...
    Ot(nframes/2 - 1 : end, :), [3], [1, 0, 0])




