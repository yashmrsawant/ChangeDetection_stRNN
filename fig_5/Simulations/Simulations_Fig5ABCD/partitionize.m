function M = partitionize(A, k)
%MAKEPARTITIONS Summary of this function goes here
% k is kth type of partitions of A out of 4 possibilities
%   Detailed explanation goes here

idxs_x1 = [1 : 8 : 1000];
idxs_y1 = [1 : 8 : 800];
idxs_x2 = [5 : 8 : 1000-4];
idxs_y2 = [5 : 8 : 800-4];
M1 = zeros(12500, 8, 8);
M2 = zeros(12500, 8, 8);
M3 = zeros(12500, 8, 8);
M4 = zeros(12500, 8, 8);
count = 1;
for i = idxs_x1
    for j = idxs_y1
        M1(count, :, :) = A(i + [0:7], j + [0:7]);
        count = count + 1;
    end
end
count = 1;
for i = idxs_x1
    for j = idxs_y2
        M2(count, :, :) = A(i + [0:7], j + [0:7]);
        count = count + 1;
    end
end
count = 1;
for i = idxs_x2
    for j = idxs_y1
        M3(count, :, :) = A(i + [0:7], j + [0:7]);
        count = count + 1;
    end
end
count = 1;
for i = idxs_x2
    for j = idxs_y2
        M4(count, :, :) = A(i + [0:7], j + [0:7]);
        count = count + 1;
    end
end

M1 = reshape(M1, 12500, 64);
M2 = reshape(M2, 12500, 64);
M3 = reshape(M3, 12500, 64);
M4 = reshape(M4, 12500, 64);
% t = toc(tstart);

if k == 1
    M = M1;
elseif k == 2
    M = M2;
elseif k == 3
    M = M3;
elseif k == 4
    M = M4;
end
%fprintf("time elapsed = %f\n", t);
end