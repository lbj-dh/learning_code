

import torch.nn
from prompt_toolkit import output


class DH(torch.nn.Module):
    def __init__(self):
        super(DH,self).__init__()

    def forward(self, input):
        output = input + 1
        return  output

dh = DH()
x = torch.tensor(1)
output = dh(x)
print(output)