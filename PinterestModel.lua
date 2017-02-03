require 'nngraph';
require 'nn';
require 'torch';
require 'LSTM';

local utils = require 'util.utils'
local PM, parent = torch.class('nn.PinterestModel', 'nn.Module')

D = 10
H = 20
C = 15
in_h0 = nn.Identity()()
in_c0 = nn.Identity()()
in1 = nn.Identity()()
in2 = nn.Identity()()
in3 = nn.Identity()()
in4 = nn.Identity()()
in5 = nn.Identity()()
model = nn.LSTM(D, H)
-- h: Sequence of hidden states, (N, T, H)
h1= model:clone('weight', 'bias', 'gradWeight', 'gradBias')({in_c0, in_h0, in1})
h1, c1 = h1 : split(2)
out1 = nn.LogSoftMax()(nn.Linear(H, C)(nn.Select(1, 1)(nn.Select(1, 1)(h1))))

h2= model:clone('weight', 'bias', 'gradWeight', 'gradBias')({nn.Select(2, 1)(c1), nn.Select(2, 1)(h1), in2})
h2, c2 = h2 : split(2)
out2 = nn.LogSoftMax()(nn.Linear(H, C)(nn.Select(1, 1)(nn.Select(1, 1)(h2))))

h3= model:clone('weight', 'bias', 'gradWeight', 'gradBias')({nn.Select(2, 1)(c1), nn.Select(2, 1)(h1), in3})
h3, c3 = h3 : split(2)
out3 = nn.LogSoftMax()(nn.Linear(H, C)(nn.Select(1, 1)(nn.Select(1, 1)(h3))))

h4= model:clone('weight', 'bias', 'gradWeight', 'gradBias')({nn.Select(2, 1)(c3), nn.Select(2, 1)(h3), in4})
h4, c4 = h4 : split(2)
out4 = nn.LogSoftMax()(nn.Linear(H, C)(nn.Select(1, 1)(nn.Select(1, 1)(h4))))

h5= model:clone('weight', 'bias', 'gradWeight', 'gradBias')({nn.Select(2, 1)(c3), nn.Select(2, 1)(h3), in5})
h5, c5 = h5 : split(2)
out5 = nn.LogSoftMax()(nn.Linear(H, C)(nn.Select(1, 1)(nn.Select(1, 1)(h5))))

mlp = nn.gModule( {in_h0, in_c0, in1, in2, in3, in4, in5}, {out1, out2, out3, out4, out5, c2, c4, c5} )

h0 = torch.randn(1, H)
c0 = torch.randn(1, H)
x1 = torch.randn(1, 1, D)
x2 = torch.randn(1, 1, D)
x3 = torch.randn(1, 1, D)
x4 = torch.randn(1, 1, D)
x5 = torch.randn(1, 1, D)

dx1 = torch.randn(C)
dx2 = torch.randn(C)
dx3 = torch.randn(C)
dx4 = torch.randn(C)
dx5 = torch.randn(C)
dc2 = torch.randn(1, 1, H)
dc4 = torch.randn(1, 1, H)
dc5 = torch.randn(1, 1, H)

forward_rlt = mlp:forward({h0, c0, x1, x2, x3, x4, x5})
print(forward_rlt)
backward_rlt = mlp:backward({h0, c0, x1, x2, x3, x4, x5}, {dx1, dx2, dx3, dx4, dx5, dc2, dc4, dc5})
print(backward_rlt)

-- draw graph (the forward graph, '.fg')
graph.dot(mlp.fg, 'MLP', 'PinterestModel')