input = imread('../images/image4.jpg');

I = rgb2gray(input);
level = graythresh(I);
BW = imbinarize(I,level);
I = 1 - BW;
I = bwareaopen(I, 5);
CCs = bwconncomp(I);
images = zeros(28*28,50);
figure()
for i=1:50
   [cols, rows] = ind2sub(size(I), CCs.PixelIdxList{i});
   img = imcrop(I, [min(rows), min(cols),max(rows)- min(rows), max(cols)- min(cols)]);
   img = padarray(img,[5,5],0);
   img = imresize(img, [28, 28]);
   subplot(5,10,i);
   imshow(img);
   img = img';
   img = img(:);
   images(:,i) = img;   
   
end

layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = true;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

data = xtest(:, 1:100);
data(:,1:50) = images;
[output, result] = convnet_forward(params, layers, data);
[~, result] = max(result);
cap = string(result(1:50)-1);
cap = join(cap,'-');
figure();
imshow(input)
title(cap, 'FontSize',10)
