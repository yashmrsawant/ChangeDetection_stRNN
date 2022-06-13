clear;
close all;
clc;
%% Display connectivity matrices
wh_fc = dlmread('fcRNN_weights_Wh40.csv');
wh_st = dlmread('stRNN_weights_Wh35.csv');

figure;
subplot(1,2,1); 
imagesc((-wh_fc),[-1,1]); colormap(redblue); title('fc-RNN');
axis square;

subplot(1,2,2); 
imagesc((-wh_st), [-1,1]); colormap(redblue); title('st-RNN');
axis square;