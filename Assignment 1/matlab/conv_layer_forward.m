function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: 

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

%% Fill in the code

outData = zeros(h_out, w_out, num, batch_size);


% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape. 
convolved = zeros(h_out, w_out, num, batch_size);

for n=1:batch_size
    
    DataInput = input.data(:,n);
    DataInput = reshape(DataInput, h_in, w_in, c);
    padded = padarray(DataInput, [pad pad]);
    
    for num=1:num
        weight1 = param.w(:,num);
        weight = reshape(weight1, k, k, c);
        
        countY = 1;
        for i = 1:stride:size(padded,1)-k+1
            countX = 1;
            for j = 1: stride: size(padded,2) - k + 1
                    matrix = padded(i:i+k-1, j:j+k-1, :);
                    outData(countY, countX, num, n)  = sum(sum(sum(matrix .* weight))) + param.b(num);
                    countX = countX+1;
            end
            countY = countY + 1;
        end

%        finalConvolved(:,:,num) = convolved;
    end
    
end
% outData= reshape(convolved, h_out*w_out*num,batch_size);

% output.data = outData;
output.data = reshape(outData, [h_out*w_out*num,batch_size]);

output.height = h_out;
output.width = w_out;
output.channel = num;
output.batch_size = batch_size;


end



