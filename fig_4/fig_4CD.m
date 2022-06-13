clear;
close all;
clc;
%% visualize joint subspace model %%
load('data_projected_pc_coeff.mat');
data_no=6; time_sam = 18;  start_step = 3; ip1_id = [0,1,2,3,4,5]+1;
C = repmat(start_step:time_sam,data_no,1); S = repmat(30*ones(1,time_sam-start_step+1),data_no,1);
cc = C(:); ss = S(:);

figure; 
%%% include plot lines %%%
hold on;
for i=1:data_no
    plot3(YY_t(i:data_no:end,1),YY_t(i:data_no:end,2), YY_t(i:data_no:end,3),'LineWidth',2); hold on;
end;
zlim([-1.5 0.6]); %grid on; 
XL = get(gca, 'XLim'); YL = get(gca, 'YLim'); ZL = get(gca, 'ZLim'); hold on;
points =  [[XL(1), XL(2), XL(2), XL(1)]; [YL(1), YL(1), YL(2), YL(2)]; ZL(1)*ones(1,4)];
h1 = fill3(points(1,:),points(2,:),points(3,:),'k'); %alpha(0.15); hold on;
points =  [[XL(1), XL(2), XL(2), XL(1)]; YL(2)*ones(1,4); [ZL(1), ZL(1), ZL(2), ZL(2)]];
h2 = fill3(points(1,:),points(2,:),points(3,:),'k'); %alpha(0.00); hold on;
points =  [ XL(2)*ones(1,4); [YL(1), YL(2), YL(2), YL(1)]; [ZL(1), ZL(1), ZL(2), ZL(2)]];
h3 = fill3(points(1,:),points(2,:),points(3,:),'k'); %alpha(0.15);
set(h1,'facealpha',.4); set(h2,'facealpha',.1); set(h3,'facealpha',.1);
xlabel('Stimulus PC1'); ylabel('Stimulus PC2'); zlabel('Time PC1'); grid off;

figure; scatter(YY_t(:,1), YY_t(:,2), ss, cc); hold on;
for i=1:data_no
    plot(YY_t(i:data_no:end,1),YY_t(i:data_no:end,2)); hold on;
    text(YY_t(i,1)-0.05,YY_t(i,2)-0.3, num2str(ip1_id(i)-1), 'FontSize',14); hold on;
end;
xlim([-0.5 4.5]); grid on; xlabel('Stimulus PC1'); ylabel('Stimulus PC2');




%% visualize input stimulus %%
load('object_samples_8.mat')
cc1 = cat(3, aa0, aa1, aa2, aa3, aa4, aa5, aa6, aa7); cc1 = permute(cc1,[3 1 2]);
target_data_current = cc1;
target_data_later = cc1;

image_sequence_1 = cell(1,4); stimulus_sequence_1 = cell(1,4);
image_sequence_2 = cell(1,4); stimulus_sequence_2 = cell(1,4);
image_sequence_3 = cell(1,4); stimulus_sequence_3 = cell(1,4);
for qq=1:3
for ii=1:4
    img = 255*ones(8*3,8*3,3);
    inp_id =[0,0,0,0,0,0,0,0,0]; 
    if ii==1, inp_id =[0,5,0,4,1,2,0,3,0];  %inp_id =[0,1,0,2,5,4,0,3,0]; 
    elseif ii==3, inp_id =[0,5,7,4,0,6,0,3,0]; end; %inp_id =[0,1,7,2,0,6,0,3,0];
    if (ii==3 && qq==3), inp_id =[0,5,7,4,1,6,0,3,0];  end; %inp_id =[0,1,7,2,5,6,0,3,0];
    kk=0;
    for j1=0:2
        for j2=0:2
            kk=kk+1;
            if qq==1, aa(j1*8+1:j1*8+8,j2*8+1:j2*8+8) = squeeze(target_data_current(inp_id(kk)+1,:,:));
            else aa(j1*8+1:j1*8+8,j2*8+1:j2*8+8) = squeeze(target_data_later(inp_id(kk)+1,:,:)); 
     end; end; end; 
    aa = double(aa>0.5);
    for i1=1:24
        for j1=1:24
            if aa(i1,j1)==1, img(i1,j1,:) = [0, 0, 255];
            else img(i1,j1,:) = [254, 254, 192];
    end;  end;  end;
    if qq==1, stimulus_sequence_1{ii} = aa; image_sequence_1{ii} = uint8(imresize(img, 10, 'nearest'));
    elseif qq==2, stimulus_sequence_2{ii} = aa; image_sequence_2{ii} = uint8(imresize(img, 10, 'nearest'));
    else stimulus_sequence_3{ii} = aa; image_sequence_3{ii} = uint8(imresize(img, 10, 'nearest'));
    end;
end; end;
figure;
ii=1; subplot(4,4,ii); imshow(image_sequence_1{ii}); title(num2str(ii));
ii=2; subplot(4,4,ii); imshow(image_sequence_1{ii}); title(num2str(ii));
ii=3; subplot(4,4,ii); imshow(image_sequence_1{ii}); title(num2str(ii));
ii=4; subplot(4,4,ii); imshow(image_sequence_1{ii}); title(num2str(ii));

ii=1; subplot(4,4,4+ii); imshow(image_sequence_3{ii}); title(num2str(ii));
ii=2; subplot(4,4,4+ii); imshow(image_sequence_3{ii-1}); title(num2str(ii));
ii=3; subplot(4,4,4+ii); imshow(image_sequence_3{ii}); title(num2str(ii));
ii=4; subplot(4,4,4+ii); imshow(image_sequence_3{ii-1}); title(num2str(ii));

ii=1; subplot(4,4,8+ii); imshow(image_sequence_2{ii}); title(num2str(ii));
ii=2; subplot(4,4,8+ii); imshow(image_sequence_2{ii-1}); title(num2str(ii));
ii=3; subplot(4,4,8+ii); imshow(image_sequence_2{ii}); title(num2str(ii));
ii=4; subplot(4,4,8+ii); imshow(image_sequence_2{ii-1}); title(num2str(ii));
subplot(4,4,15); imagesc(uint8(abs(double(image_sequence_1{1}) - double(image_sequence_1{3}))));

aaa = uint8(abs(double(image_sequence_1{1}) - double(image_sequence_1{3})));
aaa(aaa>10) = 255;






