clear;
close all;
clc;
%% Load activation of hidden neurons (EI)
load('data_hidden_activation_stRNN.mat');
hidden_s = max(hidden_s, 0);
hidden_data = zeros(500*20,320);
k=0;
for i=1:500
    for j=1:20
        k=k+1;
        hidden_data(k,:) = hidden_s(i,j,:);
    end;
end;
    
% labeling
label_data = zeros(1,500*20);
label_select = zeros(1,500*20);
k=1;
for i=1:500
    vec1 = 4*ones(1,20);
    vec_loc = zeros(1,20);
    vec2 = [1,2*ones(1,k-1), 3];
    vec1(1:size(vec2,2)) = vec2;
    if k==1
        vec_loc([k+1, randi([k+2 20],1)]) = 1;
    elseif k==19
        vec_loc([randi([2 k],1), k+1]) = 1;
    else
        vec_loc([randi([2 k],1), k+1, randi([k+2 20],1)]) = 1;
    end;
    if (k==19) k=1; else k=k+1; end;
    label_select(1, (i-1)*20+1:i*20) = vec_loc;
    label_data(1, (i-1)*20+1:i*20) = vec1;
end;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% PCA analysis for mnemonic Subspace %%%%%%%%%%%%%%%%%%%%%%%%
hidden_data_current_pre = zeros(27 ,18, 320);

for i=0:26
    hidden_data_current_pre(i+1,:,:) = hidden_data((20*i*19+1) +2 :20*i*19+20, :);
end;
hidden_data_current = reshape(hidden_data_current_pre - ...
    permute(repmat(squeeze(mean(hidden_data_current_pre, 1)), 1, 1, 27), [3,1,2]), [27*18, 320]);

[COEFF,latent,~] = pcacov((hidden_data_current'*hidden_data_current)./(27*18-1));
SCORE = hidden_data_current*COEFF;

%%%%%%%%%%%% for full hidden representation
time_sam=18; data_no=7; start_step=1;
data_set = 1+[4,5,6,9,10,18,15]; 
%data_set = 1:20
hidden_data_current_1 = reshape(hidden_data_current, [27, 18, 320]);
YY = reshape(hidden_data_current_1(data_set,[start_step,3:time_sam],:), [data_no*(time_sam-start_step+1-1), 320])*COEFF(:,1:2);
C = repmat([start_step,3:time_sam],data_no,1); S = repmat(90*ones(1,time_sam-start_step+1-1),data_no,1);
cc = C(:); ss = S(:); plot_col = ['r','g','b','m','y', 'c','k'];

figure; scatter(YY(:,1), YY(:,2),ss, cc, 'filled'); hold on;
for i=1:data_no
    plot(YY(i:data_no:end,1),YY(i:data_no:end,2),'linewidth', 2,'color', [0,0,0]+0.7 ); hold on;
end;
legend({'','5','6','7','10','11','19','16'})
xlabel('Stimulus PC1'); ylabel('Stimulus PC2'); grid off; %axis equal;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% PCA analysis for time axis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hidden_data_current_pre = zeros(27 ,18, 320);
for i=0:26
    hidden_data_current_pre(i+1,:,:) = hidden_data((20*i*19+1) +2 :20*i*19+20, :);
end;
hidden_data_current = squeeze(mean(hidden_data_current_pre, 1));

[COEFF_t_,latent_t,explained] = pcacov(cov(hidden_data_current'));
COEFF_t = (hidden_data_current' - repmat(mean(hidden_data_current, 1)', 1,18))*COEFF_t_;
COEFF_t(:,1) = COEFF_t(:,1)./norm(COEFF_t(:,1));
correlation_with_mnemonics = COEFF'*COEFF_t(:,1);
projection_on_mnemonic = (COEFF_t(:,1)'*COEFF(:,1:2)) * COEFF(:,1:2)';

pc_t = COEFF_t(:,1) - projection_on_mnemonic';
pc_t = pc_t./norm(pc_t);
COEFF'*pc_t;

projection_on_mnemonic_2 = (COEFF_t(:,2)'*[COEFF(:,1:2), pc_t]) * [COEFF(:,1:2), pc_t]';
pc_t_2 = COEFF_t(:,2) - projection_on_mnemonic_2';
pc_t_2 = pc_t_2./norm(pc_t_2);

%% %%%%%%%% Visualization of in 3D %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_no=size(data_set,2);
time_sam = 18;
start_step = 3;
YY_t = reshape(hidden_data_current_pre(data_set,start_step:time_sam,:), [data_no*(time_sam-start_step+1), 320])*[COEFF(:,1:2), pc_t];
C = repmat(start_step:time_sam,data_no,1); S = repmat(60*ones(1,time_sam-start_step+1),data_no,1);
cc = C(:); ss = S(:);

figure; scatter3(YY_t(:,1), YY_t(:,2), YY_t(:,3), ss, cc, 'filled'); hold on;
hold on;
for i=1:data_no
    plot3(YY_t(i:data_no:end,1),YY_t(i:data_no:end,2), YY_t(i:data_no:end,3),...
        'LineWidth',2, 'color', [0,0,0]+0.7); hold on;
    plot3(YY_t(i:data_no:end,1),YY_t(i:data_no:end,2), -2*ones(size(YY_t(i:data_no:end,3))),...
        'LineWidth',4, 'color', [0,0,0]+0.3); hold on;
end;

xlim([1 8]);
zlim([-2 0.6]); %grid on;
XL = get(gca, 'XLim');
YL = get(gca, 'YLim');
ZL = get(gca, 'ZLim');
hold on;
points =  [[XL(1), XL(2), XL(2), XL(1)]; [YL(1), YL(1), YL(2), YL(2)]; ZL(1)*ones(1,4)];
h1 = fill3(points(1,:),points(2,:),points(3,:),'k'); %alpha(0.15);
hold on;
points =  [[XL(1), XL(2), XL(2), XL(1)]; YL(2)*ones(1,4); [ZL(1), ZL(1), ZL(2), ZL(2)]];
h2 = fill3(points(1,:),points(2,:),points(3,:),'k'); %alpha(0.00);
hold on;
points =  [ XL(2)*ones(1,4); [YL(1), YL(2), YL(2), YL(1)]; [ZL(1), ZL(1), ZL(2), ZL(2)]];
h3 = fill3(points(1,:),points(2,:),points(3,:),'k'); %alpha(0.15);
set(h1,'facealpha',.3); set(h2,'facealpha',.1); set(h3,'facealpha',.1);
xlabel('Stimulus PC1'); ylabel('Stimulus PC2'); zlabel('Time PC1'); grid off;

%% %%%%%%%% Visualization of in 3D with two time PC dimensions %%%%%%%%%%
data_no=size(data_set,2);
time_sam = 18;
start_step = 3;
YY_t = reshape(hidden_data_current_pre(data_set,start_step:time_sam,:), [data_no*(time_sam-start_step+1), 320])*[COEFF(:,1:2), pc_t, pc_t_2];
C = repmat(start_step:time_sam,data_no,1); S = repmat(60*ones(1,time_sam-start_step+1),data_no,1);
cc = C(:); ss = S(:);
st_val = [];
figure;
for i=[5,3]
    if i==5
        scatter3(4.5*YY_t(i:data_no:end,3), YY_t(i:data_no:end,1), YY_t(i:data_no:end,4), ss(i:data_no:end), cc(i:data_no:end), 'filled'); hold on;
        plot3(4.5*YY_t(i:data_no:end,3),YY_t(i:data_no:end,1), YY_t(i:data_no:end,4),...
        'LineWidth',2, 'color', [0,0,0]+0.7); hold on;
    else
        scatter3(YY_t(i:data_no:end,3), YY_t(i:data_no:end,1), YY_t(i:data_no:end,4), ss(i:data_no:end), cc(i:data_no:end), 'filled'); hold on;
        plot3(YY_t(i:data_no:end,3),YY_t(i:data_no:end,1), YY_t(i:data_no:end,4),...
        'LineWidth',2, 'color', [0,0,0]+0.7); hold on;
    end;
    st_val = [st_val, mean(YY_t(i:data_no:end,1))];
end;
xlabel('Time PC1'); ylabel('Stimulus PC'); zlabel('Time PC2');

xlim([-1.4 0.3]); zlim([-0.6 1.1]); ylim([-1 8.5]); %grid on;
XL = get(gca, 'XLim'); YL = get(gca, 'YLim'); ZL = get(gca, 'ZLim'); hold on;
points =  [[XL(1), XL(2), XL(2), XL(1)]; st_val(1)+0.2*ones(1,4); [ZL(1), ZL(1), ZL(2), ZL(2)]];
h1_ = fill3(points(1,:),points(2,:),points(3,:),'k'); hold on; %alpha(0.00); 

points =  [[XL(1), XL(2), XL(2), XL(1)]; st_val(2)*ones(1,4); [ZL(1), ZL(1), ZL(2), ZL(2)]];
h2_ = fill3(points(1,:),points(2,:),points(3,:),'k'); hold on; %alpha(0.00);
set(h1_,'facealpha',.07); set(h2_,'facealpha',.07);

points =  [[XL(1), XL(2), XL(2), XL(1)]; [YL(1), YL(1), YL(2), YL(2)]; ZL(1)*ones(1,4)];
h1 = fill3(points(1,:),points(2,:),points(3,:),'k'); hold on; %alpha(0.15);

points =  [ XL(2)*ones(1,4); [YL(1), YL(2), YL(2), YL(1)]; [ZL(1), ZL(1), ZL(2), ZL(2)]];
h3 = fill3(points(1,:),points(2,:),points(3,:),'k'); %alpha(0.15);
set(h1,'facealpha',.01); set(h3,'facealpha',.01); %set(h2,'facealpha',.0);
grid off;
