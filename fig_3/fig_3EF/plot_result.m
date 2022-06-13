function plot_result(Xt, Yt, Ot, ts, color)
%% plotting result
n_timesteps = size(Xt, 1);
for i = 1 : n_timesteps
    subplot(3, n_timesteps, i);
    img = imresize(reshape(Xt(i, :), 8, 8), ...
                [10, 10], 'nearest');
   
    imshow(1-img);
    sz_q = size(img, 2); sz_p = size(img, 1);
    if ~isempty(find(ts == i))
        rectangle('Position', [0.5, 0.5, sz_q, sz_p], 'EdgeColor', color);
    end
    if i == 1
        title("I/P at t = 1");
    else
        title(strcat("t = ", num2str(i)));
    end
    
    subplot(3, n_timesteps, i+n_timesteps);
    img = imresize(reshape(Yt(i, :), 8, 8), ...
                [10, 10], 'nearest');
    imshow(1-img);
    if i == 1
        title("Exp. at t = 1"); % Expected
    end
    subplot(3, n_timesteps, i+2*n_timesteps);
    img = imresize(reshape(Ot(i, :), 8, 8), ...
                [10, 10], 'nearest');
    imshow(1-img);
    if i == 1
        title("Obs. at t = 1"); % Observed
    end
    %hp4 = get(subplot(3, n_timesteps, n_timesteps), 'Position');
    %h = colorbar('Position', [hp4(1)+hp4(3)+0.02 hp4(2) 0.01 hp4(3)*5.1]);
    %set(h, 'YDir', 'reverse');
end
end