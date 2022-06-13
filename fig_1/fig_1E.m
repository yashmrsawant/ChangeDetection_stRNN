clear;
close all;
clc;
%% 
wx = dlmread('loss_mc_fcrnn.csv');
plot(wx(1:200))
hold on;
wx = dlmread('loss_mc_strnn.csv');
plot(wx(1:200))
xlabel('Iteration')
ylabel('MSE error')
title('Mnemonic-coding RNN')
legend('st-RNN','fc-RNN');
grid off; box off;

%% 
figure;
wx = dlmread('loss_cd_strnn.csv');
plot(wx(1:20:4000))
hold on;
wx = dlmread('loss_cd_fcrnn.csv');
plot(wx(1:20:4000))
xlabel('Iteration')
ylabel('MSE error')
title('Change-detection RNN')
legend('st-RNN','fc-RNN');
grid off;
box off;