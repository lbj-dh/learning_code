import torch
from d2l import torch as d2l
from torch import nn
from matplotlib import pyplot as plt



n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01,0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


def init_params():
    w = torch.normal(0,1,size=(num_inputs, 1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w,b = init_params()
    net,loss = lambda X: d2l.linreg(X,w,b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale= 'log', xlim=
                            [5,num_epochs],legend=['train','test'])
    for epoch in range(num_epochs):
        for X, Y in train_iter:
            l = loss(net(X), Y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w,b], lr,batch_size)
        if (epoch+1) % 5 == 0:
            animator.add(epoch+1,(d2l.evaluate_loss(net,train_iter,loss)
                         ,d2l.evaluate_loss(net,test_iter,loss)))
    print('w的L2范数是:',torch.norm(w).item())

train(lambd =  0)

train(lambd = 3)
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    trainer = torch.optim.SGD([
        {'params': net[0].weight, 'weight_decay': wd},
        {'params': net[0].bias}
    ],lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=
                            [5,num_epochs],legend=['train','test'])

    for epoch in range(num_epochs):
        for X, Y in train_iter:
            trainer.zero_grad()
            l = loss(net(X),Y)
            l.mean().backward()
            trainer.step()
        if (epoch+1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss)
                                         , d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是:', net[0].weight.norm().item())
train_concise(0)
train_concise(3)
plt.show()

