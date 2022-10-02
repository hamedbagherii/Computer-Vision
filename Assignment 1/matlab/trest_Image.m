layers = get_lenet();
load lenet.mat
fullset = true;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
images = zeros(28*28,5);
folder = '../images/';
for i =1:5
   Image = dir(sprintf('../images/%d.jpg',i)).name;
    I=im2double(imread(fullfile(folder,Image)));  
    I = rgb2gray(I);
    I = imresize(I,[28,28]);
    level = graythresh(I);
    I = 1 - I;
    I = I';
    I = I(:);
    
    images(:,i) = I;

data = xtest(:, 1:1000);
data(:,1:5) = images;
[output, result] = convnet_forward(params, layers, data);
[~, result] = max(result);
for i=1:5
    Image = dir(sprintf('../images/%d.jpg',i)).name;
    I=imread(fullfile(folder,Image)); 
    subplot(1,5,i);
    imshow(I);
    t=title(result(1,i)-1);
    t.Color = 'red';
    t.FontSize=25;

end
end
