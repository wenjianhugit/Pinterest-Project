require 'nngraph';
require 'nn';
require 'torch';
require 'LSTM';

D = 10
H = 10
C = 10

inin = nn.Identity()()
input = nn.SplitTable(1)(inin)
in_c0 = nn.Reshape(1, H)(nn.SelectTable(1)(input))
in_h0 = nn.Reshape(1, H)(nn.SelectTable(2)(input))
in1 = nn.Reshape(1, 1, D)(nn.SelectTable(3)(input))
in2 = nn.Reshape(1, 1, D)(nn.SelectTable(4)(input))
in3 = nn.Reshape(1, 1, D)(nn.SelectTable(5)(input))
in4 = nn.Reshape(1, 1, D)(nn.SelectTable(6)(input))
in5 = nn.Reshape(1, 1, D)(nn.SelectTable(7)(input))
model = nn.LSTM(D, H)
-- h: Sequence of hidden states, (N, T, H)
h1= model:clone('weight', 'bias', 'gradWeight', 'gradBias')({in_c0, in_h0, in1})
h1, c1 = h1 : split(2)

h2= model:clone('weight', 'bias', 'gradWeight', 'gradBias')({nn.Select(2, 1)(c1), nn.Select(2, 1)(h1), in2})
h2, c2 = h2 : split(2)

h3= model:clone('weight', 'bias', 'gradWeight', 'gradBias')({nn.Select(2, 1)(c1), nn.Select(2, 1)(h1), in3})
h3, c3 = h3 : split(2)

h4= model:clone('weight', 'bias', 'gradWeight', 'gradBias')({nn.Select(2, 1)(c3), nn.Select(2, 1)(h3), in4})
h4, c4 = h4 : split(2)

h5= model:clone('weight', 'bias', 'gradWeight', 'gradBias')({nn.Select(2, 1)(c3), nn.Select(2, 1)(h3), in5})
h5, c5 = h5 : split(2)

out = nn.JoinTable(3)({h1, h2, h3, h4, h5, c2, c4, c5})
mlp = nn.gModule( {inin}, {out} )

x = torch.randn(7, H)
dout = torch.randn(1, 1, 8*H)

forward_rlt = mlp:forward(x)
print(forward_rlt)
backward_rlt = mlp:backward(x, dout)
print(backward_rlt)

local precision = 1e-5
jac = nn.Jacobian
err = jac.testJacobian(mlp, x, 0, 1, 1e-6)
print('==> error: ' .. err)
if err<precision then
	print('==> module OK')
else
  	print('==> error too large, incorrect implementation')
end