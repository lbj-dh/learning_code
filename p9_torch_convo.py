import torch
import torch.nn.functional as F
input_torch = torch.tensor([
    [1,2,0,3,1],
    [0,1,2,3,1],
    [1,2,1,0,0],
    [5,2,3,1,1],
    [2,1,0,1,1]
])

con_torch = torch.tensor([
    [1,2,1],
    [0,1,0],
    [2,1,0]
])

input_torch = torch.reshape(input_torch, (1,1,5,5))
con_torch = torch.reshape(con_torch, (1,1,3,3))

result_con = F.conv2d(input_torch, con_torch,stride=1,padding=1)

print(result_con)


