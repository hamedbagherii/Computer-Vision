%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    [~, labels] = max(P);
    for j=0:99
        label = labels(j+1);
        GT = ytest(i+j);
        confusion(GT, label) = confusion(GT, label) +1;
        
    end
end