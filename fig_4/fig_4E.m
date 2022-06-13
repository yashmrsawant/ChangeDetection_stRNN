clear;
close all;
clc;
load('data_hidden_activation_stRNN.mat');
%% Raw neural activity after global inhibition
stim_id_odd = 7;
stim_id_even = stim_id_odd+1;
std_nid_320 = mean(squeeze(std(hidden_n_9(stim_id_odd:stim_id_odd+1,4:10,:),0,2)),1);
[~, nid_sort] = sort(std_nid_320,'descend');

figure;
for i=1:5
    plot( hidden_n_9(stim_id_odd,:,nid_sort(i))', 'LineWidth',4); hold on;
end; 
xlabel("Time");
ylabel("Activation");

figure;
for i=1:5
    plot( hidden_n_9(stim_id_even,:,nid_sort(i))', 'LineWidth',4); hold on;
end; 
axis([0 inf -0.05 1.05])
xlabel("Time");
ylabel("Activation");