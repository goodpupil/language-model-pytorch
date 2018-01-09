from model import CharCNN
from torch.autograd import Variable
import torch

charcnn = CharCNN(5, 30, 60)
test_input = Variable(torch.randn(2, 5, 30))

print(charcnn(test_input))