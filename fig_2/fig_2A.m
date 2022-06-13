clear;
close all;
clc;
%% Load saved neural activity at the hidden layer for different sparsity level
load_eigen_vec=1;
load('data_w_varied_sparsity_hidden_activation_stRNN.mat');

hidden_s = max(hidden_s, 0);
hidden_data = zeros(5000*40,320);
k=0;
for i=1:5000
    for j=1:40
        k=k+1;
        hidden_data(k,:) = hidden_s(i,j,:);
    end;
end;
%[COEFF,SCORE,latent] = princomp(hidden_data);
    
% labeling as in hidden_data
label_data = zeros(1,5000*40);
label_select = zeros(1,5000*40);
label_time_avg = zeros(1, 5000*40);
label_first_4 = zeros(1, 5000*40);
k=1; count=0;
for i=1:5000
    vec1 = 4*ones(1,20);
    vec_loc = zeros(1,20);
    vec_first_4 = zeros(1,20);
    vec2 = [1,2*ones(1,k-1), 3];
    vec1(1:size(vec2,2)) = vec2;
    % to remove first 4 values
    if k>=16 
        vec1(1:size(vec2,2)+19-k)= 0; vec1(1:size(vec2,2)) = vec2;
    else vec1(1:size(vec2,2)+4)= [vec2, 0, 0, 0, 0]; end;
    if k<16-5 
        count=count +1; vec_first_4(length([vec2, 0, 0, 0, 0])+1)=1;
        label_first_4(1, (i-1)*40+1:i*40) = [vec_first_4, zeros(1,20)];
        label_time_avg(1, (i-1)*40+1:i*40) = [(vec1==4)*count, ones(1,20)*count];
    end;
    % end
    if k==1
        vec_loc([k+1, randi([k+2 20],1)]) = 1;
    elseif k==19
        vec_loc([randi([2 k],1), k+1]) = 1;
    else
        vec_loc([randi([2 k],1), k+1, randi([k+2 20],1)]) = 1;
    end;
    if (k==19) k=1; else k=k+1; end;
    label_select(1, (i-1)*20+1:i*20) = vec_loc;
    label_data(1, (i-1)*40+1:i*40) = [vec1, 4*ones(1,20)];
end;
reshape(label_data, 40, 5000);
reshape(label_time_avg, 40, 5000);
reshape(label_first_4, 40, 5000);

%% Generate separate data for computing stimulus PC vs. time PC
label_unique_data =  reshape(repmat(1:5000, 40, 1), 1, 40*5000);
hidden_data_current_more_t = zeros(sum(label_time_avg>0), 320);
ii=0;
for i=1:count
    mat_temp = hidden_data(label_time_avg==i,:)  - ...
        repmat(mean(hidden_data(label_time_avg==i, :)), sum(label_time_avg==i), 1);
    hidden_data_current_more_t(ii+1:ii+sum(label_time_avg==i), :) = mat_temp;
    ii = ii+sum(label_time_avg==i);
end;

hidden_data_current_more = zeros(sum(label_first_4), 320);
for i=1:sum(label_first_4)
    hidden_data_current_more(i,:) = hidden_data((label_time_avg==i & label_first_4==1), :);
end;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% PCA analysis for mnemonic Subspace %%%%%%%%%%%%%%%%%%%%%%%%
size_case_1 = length(1:19:5000);
hidden_data_current_pre = zeros(size_case_1 ,18+20, 320);
label_first_4_current = zeros(size_case_1 ,18+20);
for i=0:size_case_1-1
    hidden_data_current_pre(i+1,:,:) = hidden_data((40*i*19+1) +2 :40*i*19+40, :);
    label_first_4_current(i+1,:) = label_first_4((40*i*19+1) +2 :40*i*19+40);
end;
label_first_4_current = reshape(label_first_4_current, size_case_1*(18+20), 1);

hidden_data_current = reshape(hidden_data_current_pre,[size_case_1*(18+20), 320]);

if load_eigen_vec==1 
    load('saved_eigen_vectors_35_t.mat');
else
    [COEFF,latent,explained] = pcacov(cov(hidden_data_current_more));
    [COEFF_t_,latent_t,explained] = pcacov(cov(hidden_data_current_more_t));
end;
aa = (COEFF_t_(:,1)'*COEFF); aa(1:2) = 0;
bb = aa*COEFF'; bb = bb/norm(bb);
pc_t = bb';

SCORE = hidden_data_current*[COEFF(:,1:2), pc_t];

%%%%%%%%%%%% grouping %%%%%%%%%%%%%%%%%%%%%
group_index_case_1 = ceil((1:19:5000) / 1001);
group_index_full = repmat(group_index_case_1, 1,18+20);

%%%%%%%%%%%% for full hidden representation
time_sam=18+20; data_no=size_case_1;
YY = SCORE(:,1:3);
figure; gscatter(YY(label_first_4_current==1,1),YY(label_first_4_current==1,2)...
    ,group_index_full(label_first_4_current==1),'rgbmk','o*s+^'); 
