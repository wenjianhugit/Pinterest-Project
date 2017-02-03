require 'nngraph';
require 'nn';
require 'torch';
require 'LSTM';

function numerical_gradient_input(c0, h0, x0, grad_h, grad_c)
	h=1e-5
	N = 1
	T = 1
	D = 10
	H = 20
	grad_h_c0 = torch.Tensor(c0:size()):zero()
	grad_c_c0 = torch.Tensor(c0:size()):zero()
	grad_c0 = torch.Tensor(c0:size()):zero()
	size = c0:size()
	for x1 = 1, size[1] do
		for x2 = 1, size[2] do 
			oldval = c0[x1][x2]

			model = nn.LSTM(D, H)
			model.weight = torch.reshape(torch.linspace(-0.9, 0.9, 2400),torch.LongStorage{30, 80})
			model.bias = torch.linspace(-0.3, 0.3, 80)
			c0[x1][x2]= oldval + h
			forward_rlt=model:forward({c0, h0, x0})
			out_h_pos, out_c_pos = table.unpack(forward_rlt)

			model = nn.LSTM(D, H)
			model.weight = torch.reshape(torch.linspace(-0.9, 0.9, 2400),torch.LongStorage{30, 80})
			model.bias = torch.linspace(-0.3, 0.3, 80)
			c0[x1][x2]= oldval - h
			forward_rlt=model:forward({c0, h0, x0})
			out_h_neg, out_c_neg = table.unpack(forward_rlt)

			c0[x1][x2]= oldval 
			grad_h_c0[x1][x2] = torch.dot((out_h_pos - out_h_neg)/(2*h), grad_h)
			grad_c_c0[x1][x2] = torch.dot((out_c_pos - out_c_neg)/(2*h), grad_c)
			grad_c0[x1][x2] = grad_h_c0[x1][x2] + grad_c_c0[x1][x2]
		end
	end

	grad_h_h0 = torch.Tensor(h0:size()):zero()
	grad_c_h0 = torch.Tensor(h0:size()):zero()
	grad_h0 = torch.Tensor(h0:size()):zero()
	size = h0:size()
	for x1 = 1, size[1] do
		for x2 = 1, size[2] do 
			oldval = h0[x1][x2]

			model = nn.LSTM(D, H)
			model.weight = torch.reshape(torch.linspace(-0.9, 0.9, 2400),torch.LongStorage{30, 80})
			model.bias = torch.linspace(-0.3, 0.3, 80)
			h0[x1][x2]= oldval + h
			forward_rlt=model:forward({c0, h0, x0})
			out_h_pos, out_c_pos = table.unpack(forward_rlt)

			model = nn.LSTM(D, H)
			model.weight = torch.reshape(torch.linspace(-0.9, 0.9, 2400),torch.LongStorage{30, 80})
			model.bias = torch.linspace(-0.3, 0.3, 80)
			h0[x1][x2]= oldval - h
			forward_rlt=model:forward({c0, h0, x0})
			out_h_neg, out_c_neg = table.unpack(forward_rlt)

			h0[x1][x2]= oldval 
			grad_h_h0[x1][x2] = torch.dot((out_h_pos - out_h_neg)/(2*h), grad_h)
			grad_c_h0[x1][x2] = torch.dot((out_c_pos - out_c_neg)/(2*h), grad_c)
			grad_h0[x1][x2] = grad_h_h0[x1][x2] + grad_c_h0[x1][x2]
		end
	end

	grad_h_x0 = torch.Tensor(x0:size()):zero()
	grad_c_x0 = torch.Tensor(x0:size()):zero()
	grad_x0 = torch.Tensor(x0:size()):zero()
	size = x0:size()
	for x1 = 1, size[1] do
		for x2 = 1, size[2] do 
			for x3 = 1, size[3] do 
				oldval = x0[x1][x2][x3]

				model = nn.LSTM(D, H)
				model.weight = torch.reshape(torch.linspace(-0.9, 0.9, 2400),torch.LongStorage{30, 80})
				model.bias = torch.linspace(-0.3, 0.3, 80)
				x0[x1][x2][x3] = oldval + h
				forward_rlt=model:forward({c0, h0, x0})
				out_h_pos, out_c_pos = table.unpack(forward_rlt)

				model = nn.LSTM(D, H)
				model.weight = torch.reshape(torch.linspace(-0.9, 0.9, 2400),torch.LongStorage{30, 80})
				model.bias = torch.linspace(-0.3, 0.3, 80)
				x0[x1][x2][x3]= oldval - h
				forward_rlt=model:forward({c0, h0, x0})
				out_h_neg, out_c_neg = table.unpack(forward_rlt)

				x0[x1][x2][x3]= oldval 
				grad_h_x0[x1][x2][x3] = torch.dot((out_h_pos - out_h_neg)/(2*h), grad_h)
				grad_c_x0[x1][x2][x3] = torch.dot((out_c_pos - out_c_neg)/(2*h), grad_c)
				grad_x0[x1][x2][x3] = grad_h_x0[x1][x2][x3] + grad_c_x0[x1][x2][x3]
			end
		end
	end

	return {grad_c0, grad_h0, grad_x0}
end

N = 1
T = 1
D = 10
H = 20
model = nn.LSTM(D, H)
model.weight = torch.reshape(torch.linspace(-0.9, 0.9, 2400),torch.LongStorage{30, 80})
model.bias = torch.linspace(-0.3, 0.3, 80)

c0 = torch.reshape(torch.linspace(-0.5, 0.4, N*H),torch.LongStorage{N, H})
h0 = torch.reshape(torch.linspace(-0.6, 0.5, N*H),torch.LongStorage{N, H})
x0 = torch.reshape(torch.linspace(-0.3, 0.3, N*T*D),torch.LongStorage{N, T, D})
grad_h = torch.reshape(torch.linspace(-1.1, 1.0, N*T*H), torch.LongStorage{N, T, H})
grad_c = torch.reshape(torch.linspace(-1.2, 1.1, N*T*H), torch.LongStorage{N, T, H})

forward_rlt=model:forward({c0, h0, x0})
backward_rlt = model:backward({c0, h0, x0}, {grad_h, grad_c})

grad_c0_num, grad_h0_num, grad_x0_num = table.unpack(numerical_gradient_input(c0, h0, x0, grad_h, grad_c))
grad_c0, grad_h0, grad_x0 = table.unpack(backward_rlt)
print(grad_c)
print(grad_c0_num)
print(grad_c0)
print('|dc0_num - dc0|^2:')
print((grad_c0 - grad_c0_num):dot(grad_c0 - grad_c0_num))
print('|dh0_num - dh0|^2:')
print((grad_h0 - grad_h0_num):dot(grad_h0 - grad_h0_num))
print('|dx0_num - dx0|^2:')
print((grad_x0 - grad_x0_num):dot(grad_x0 - grad_x0_num))
