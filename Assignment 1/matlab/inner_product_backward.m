function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

% Replace the following lines with your implementation.
param_grad.b = zeros(size(param.b));
param_grad.w = zeros(size(param.w));
input_od = param.w * output.diff;
param_grad.w =(output.diff * input.data')';
param_grad.b = sum(output.diff');
end
