
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxNorm(object):
    def __init__(self, max_value=1.0, axis=-1, epsilon=1.0e-8):
        self.max_value = max_value
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, module):
        if hasattr(module, 'weight'):
            # print("Entered for testing!")
            # w=module.weight.data
            # w=w.clamp(0.5,0.7) #将参数范围限制到0.5-0.7之间
            # module.weight.data=w
            w = module.weight.data
            # print(w.shape)
            norms = torch.norm(w, p=2, dim=self.axis, keepdim=True)
            desired = norms.clamp(0, self.max_value)
            w = w * (desired / (self.epsilon + norms))
            module.weight.data = w


"""
 Pytorch如何约束和限制权重/偏执的范围
方法一：

首先编写模型结构：

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1=nn.Linear(100,50)
        self.l2=nn.Linear(50,10)
        self.l3=nn.Linear(10,1)
        self.sig=nn.Sigmoid()
    
    def forward(self,x):
        x=self.l1(x)
        x=self.l2(x)
        x=self.l3(x)
        x=self.sig(x)
        return(x)

然后编写限制权重范围的类：

class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            print("Entered")
            w=module.weight.data
            w=w.clamp(0.5,0.7) #将参数范围限制到0.5-0.7之间
            module.weight.data=w

最后实例化这个类，对权重进行限制：

# Applying the constraints to only the last layer
constraints=weightConstraint()
model=Model()

#for i in .....模型训练代码这里请自己补充，

loss = criterion(out, var_y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

model._modules['l3'].apply(constraints)

方法二：

在模型train的时候，对参数的范围进行限制：

loss = criterion(out, var_y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

for p in net.parameters():
    p.data.clamp_(0, 99)

将权重和偏执的范围限制到0-99之间。

仅限制权重的范围，不限制偏执的范围：

for p in net.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, 0,10))

"""
