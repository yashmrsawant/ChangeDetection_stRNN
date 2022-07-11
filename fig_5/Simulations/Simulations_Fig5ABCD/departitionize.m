function A_hat = departitionize(M, k)
    idxs_x1 = [1 : 8 : 1000];
    idxs_y1 = [1 : 8 : 800];
    idxs_x2 = [5 : 8 : 1000-4];
    idxs_y2 = [5 : 8 : 800-4];
    A_hat = zeros(1000, 800);

    M = reshape(M, 12500, 8, 8);
    if k == 1
        count = 1;
        for i = idxs_x1
            for j = idxs_y1
                A_hat(i + [0:7], j + [0:7]) = A_hat(i + [0:7], j + [0:7]) + squeeze(M(count, :, :));
                count = count + 1;
            end
        end
    elseif k == 2
        count = 1;
        for i = idxs_x1
            for j = idxs_y2
                A_hat(i + [0:7], j + [0:7]) = A_hat(i + [0:7], j + [0:7]) + squeeze(M(count, :, :));
                count = count + 1;
            end
        end
    elseif k == 3
        count = 1;
        for i = idxs_x2
            for j = idxs_y1
                A_hat(i + [0:7], j + [0:7]) = A_hat(i + [0:7], j + [0:7]) + squeeze(M(count, :, :));
                count = count + 1;
            end
        end
    elseif k == 4
        count = 1;
        for i = idxs_x2
            for j = idxs_y2
                A_hat(i + [0:7], j + [0:7]) = A_hat(i + [0:7], j + [0:7]) + squeeze(M(count, :, :));
                count = count + 1;
            end
        end
    end
end


