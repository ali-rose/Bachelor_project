1.	#导库
2.	%matplotlib inline
3.	import torch
4.	import torchvision
5.	from torch.utils import data
6.	from torchvision import transforms
7.	from d2l import torch as d2l
8.	from torch import nn
9.	from torchvision import datasets
10.	from torch.optim import lr_scheduler
11.	import math
12.	#定义学习率衰减是所需要的训练函数，因为d2l库中提供的train_ch3函数只能有六个参数，所以会导致在导入学习率调度器的过程中遇到困难
13.	def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
14.	          scheduler=None):
15.	    net.to(device)
16.	    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
17.	                            legend=['train loss', 'train acc', 'test acc'])
18.
19.	    for epoch in range(num_epochs):
20.	        metric = d2l.Accumulator(3) # train_loss,train_acc,num_examples
21.	        for i, (X, y) in enumerate(train_iter):
22.	            net.train()
23.	            trainer.zero_grad()
24.	            X, y = X.to(device), y.to(device)
25.	            y_hat = net(X)
26.	            l = loss(y_hat, y)
27.	            l = torch.sum(l)
28.	            l.backward()
29.	            trainer.step()
30.	            with torch.no_grad():
31.	                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
32.	            train_loss = metric[0] / metric[2]
33.	            train_acc = metric[1] / metric[2]
34.	            if (i + 1) % 50 == 0:
35.	                animator.add(epoch + i / len(train_iter),
36.	                             (train_loss, train_acc, None))
37.
38.	        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
39.	        animator.add(epoch+1, (None, None, test_acc))
40.
41.	        if scheduler:
42.	            if scheduler.__module__ == lr_scheduler.__name__:
43.	                # UsingPyTorchIn-Builtscheduler
44.	                scheduler.step()
45.	            else:
46.	                # Usingcustomdefinedscheduler
47.	                for param_group in trainer.param_groups:
48.	                    param_group['lr'] = scheduler(epoch)
49.	    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
50.	          f'test acc {test_acc:.3f}')
51.	#网络结构
52.	net = nn.Sequential(nn.Flatten(),
53.	                    nn.Linear(784, 256),
54.	                    nn.ReLU(),
55.	                    nn.Linear(256,128),
56.	                    nn.ReLU(),
57.	                    nn.Linear(128, 10))
58.	#初始化
59.	def init_normal(m):
60.	    if type(m) == nn.Linear:
61.	        nn.init.normal_(m.weight, mean=0, std=0.01)
62.	        nn.init.zeros_(m.bias)
63.	net.apply(init_normal)
64.	#训练参数
65.	batch_size, lr, num_epochs = 256, 0.1, 10
66.	#softmax+交叉熵损失
67.	loss = nn.CrossEntropyLoss(reduction='none')
68.	trainer = torch.optim.SGD(net.parameters(), lr=lr)
69.	#多层感知机的训练
70.	train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
71.	d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
72.	#预测
73.	d2l.predict_ch3(net, test_iter)
74.	#调整输入参数方便后续
75.	def sgd(params, states, hyperparams):
76.	    for p in params:
77.	        p.data.sub_(hyperparams['lr'] * p.grad)
78.	        p.grad.data.zero_()
79.	def get_fashion_mnist_data(batch_size=256, n=1500):
80.	    # 定义图像预处理操作，将图像转换为张量，并进行标准化
81.	    transform = transforms.Compose([
82.	        transforms.ToTensor(),
83.	        transforms.Normalize((0.5,), (0.5,))
84.	    ])
85.
86.	    # 加载 Fashion MNIST 数据集
87.	    data = datasets.FashionMNIST(root='E:/Deeplearning/data/FashionMNIST/raw', train=True, transform=transform, download=True)
88.
89.	    # 假设你想使用前 `n` 个样本
90.	    data = torch.utils.data.Subset(data, list(range(n)))
91.
92.	    # 创建 DataLoader 对象，用于批量加载数据
93.	    data_iter = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
94.
95.	    return data_iter  # 假设图像是 28x28 的灰度图像
96.	#SGD优化函数
97.	def train_sgd(lr, batch_size, num_epochs):
98.	    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
99.	    return d2l.train_ch3(net, train_iter=train_iter, test_iter=test_iter, loss=loss, num_epochs=num_epochs, updater=optimizer)
100.	#调整学习率及批次
101.	gd_res = train_sgd(0.001, 1500, 10)
102.	sgd_res = train_sgd(0.005, 1,10)
103.	mini1_res = train_sgd(.4, 100,10)
104.	mini2_res = train_sgd(.05, 10,10)
105.	#固定学习率
106.	lr, num_epochs = 0.3, 10
107.	trainer = torch.optim.SGD(net.parameters(), lr=lr)
108.	device='cpu'
109.	train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
110.	#指数衰减
111.	class SquareRootScheduler:
112.	    def __init__(self, lr=0.1):
113.	        self.lr = lr
114.
115.	    def __call__(self, num_update):
116.	        return self.lr * pow(num_update + 1.0, -0.5)
117.	scheduler = SquareRootScheduler(lr=0.1)
118.	d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
119.	trainer = torch.optim.SGD(net.parameters(), lr)
120.	device='cpu'
121.	train(net, train_iter, test_iter, num_epochs, loss, trainer, device,scheduler)
122.	#多项式衰减
123.	trainer = torch.optim.SGD(net.parameters(), lr=0.5)
124.	scheduler2 = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
125.
126.	def get_lr(trainer, scheduler):
127.	    lr = scheduler2.get_last_lr()[0]
128.	    trainer.step()
129.	    scheduler2.step()
130.	    return lr
131.	d2l.plot(torch.arange(num_epochs), [get_lr(trainer, scheduler2)
132.	                                  for t in range(num_epochs)])
133.	train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
134.	    scheduler2)
135.	#余弦衰减
136.	class CosineScheduler:
137.	    def __init__(self, max_update, base_lr=0.01, final_lr=0,
138.	                warmup_steps=0, warmup_begin_lr=0):
139.	        self.base_lr_orig = base_lr
140.	        self.max_update = max_update
141.	        self.final_lr = final_lr
142.	        self.warmup_steps = warmup_steps
143.	        self.warmup_begin_lr = warmup_begin_lr
144.	        self.max_steps = self.max_update - self.warmup_steps
145.
146.	    def get_warmup_lr(self, epoch):
147.	        increase = (self.base_lr_orig - self.warmup_begin_lr) \
148.	                      * float(epoch) / float(self.warmup_steps)
149.	        return self.warmup_begin_lr + increase
150.
151.	    def __call__(self, epoch):
152.	        if epoch < self.warmup_steps:
153.	            return self.get_warmup_lr(epoch)
154.	        if epoch <= self.max_update:
155.	            self.base_lr = self.final_lr + (
156.	                self.base_lr_orig - self.final_lr) * (1 + math.cos(
157.	                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
158.	        return self.base_lr
159.
160.	scheduler3 = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
161.	d2l.plot(torch.arange(num_epochs), [scheduler3(t) for t in range(num_epochs)])
162.	trainer = torch.optim.SGD(net.parameters(), lr=0.3)
163.	train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
164.	      scheduler3)
165.	#预热
166.	scheduler4 = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
167.	d2l.plot(torch.arange(num_epochs), [scheduler4(t) for t in range(num_epochs)])
168.	trainer = torch.optim.SGD(net.parameters(), lr=0.3)
169.	train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
170.	      scheduler4)
1.	#导库
2.	%matplotlib inline
3.	import torch
4.	import torchvision
5.	from torch.utils import data
6.	from torchvision import transforms
7.	from d2l import torch as d2l
8.	from torch import nn
9.	from torchvision import datasets
10.	from torch.optim import lr_scheduler
11.	import math
12.	#定义学习率衰减是所需要的训练函数，因为d2l库中提供的train_ch3函数只能有六个参数，所以会导致在导入学习率调度器的过程中遇到困难
13.	def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
14.	          scheduler=None):
15.	    net.to(device)
16.	    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
17.	                            legend=['train loss', 'train acc', 'test acc'])
18.
19.	    for epoch in range(num_epochs):
20.	        metric = d2l.Accumulator(3) # train_loss,train_acc,num_examples
21.	        for i, (X, y) in enumerate(train_iter):
22.	            net.train()
23.	            trainer.zero_grad()
24.	            X, y = X.to(device), y.to(device)
25.	            y_hat = net(X)
26.	            l = loss(y_hat, y)
27.	            l = torch.sum(l)
28.	            l.backward()
29.	            trainer.step()
30.	            with torch.no_grad():
31.	                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
32.	            train_loss = metric[0] / metric[2]
33.	            train_acc = metric[1] / metric[2]
34.	            if (i + 1) % 50 == 0:
35.	                animator.add(epoch + i / len(train_iter),
36.	                             (train_loss, train_acc, None))
37.
38.	        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
39.	        animator.add(epoch+1, (None, None, test_acc))
40.
41.	        if scheduler:
42.	            if scheduler.__module__ == lr_scheduler.__name__:
43.	                # UsingPyTorchIn-Builtscheduler
44.	                scheduler.step()
45.	            else:
46.	                # Usingcustomdefinedscheduler
47.	                for param_group in trainer.param_groups:
48.	                    param_group['lr'] = scheduler(epoch)
49.	    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
50.	          f'test acc {test_acc:.3f}')
51.	#网络结构
52.	net = nn.Sequential(nn.Flatten(),
53.	                    nn.Linear(784, 256),
54.	                    nn.ReLU(),
55.	                    nn.Linear(256,128),
56.	                    nn.ReLU(),
57.	                    nn.Linear(128, 10))
58.	#初始化
59.	def init_normal(m):
60.	    if type(m) == nn.Linear:
61.	        nn.init.normal_(m.weight, mean=0, std=0.01)
62.	        nn.init.zeros_(m.bias)
63.	net.apply(init_normal)
64.	#训练参数
65.	batch_size, lr, num_epochs = 256, 0.1, 10
66.	#softmax+交叉熵损失
67.	loss = nn.CrossEntropyLoss(reduction='none')
68.	trainer = torch.optim.SGD(net.parameters(), lr=lr)
69.	#多层感知机的训练
70.	train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
71.	d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
72.	#预测
73.	d2l.predict_ch3(net, test_iter)
74.	#调整输入参数方便后续
75.	def sgd(params, states, hyperparams):
76.	    for p in params:
77.	        p.data.sub_(hyperparams['lr'] * p.grad)
78.	        p.grad.data.zero_()
79.	def get_fashion_mnist_data(batch_size=256, n=1500):
80.	    # 定义图像预处理操作，将图像转换为张量，并进行标准化
81.	    transform = transforms.Compose([
82.	        transforms.ToTensor(),
83.	        transforms.Normalize((0.5,), (0.5,))
84.	    ])
85.
86.	    # 加载 Fashion MNIST 数据集
87.	    data = datasets.FashionMNIST(root='E:/Deeplearning/data/FashionMNIST/raw', train=True, transform=transform, download=True)
88.
89.	    # 假设你想使用前 `n` 个样本
90.	    data = torch.utils.data.Subset(data, list(range(n)))
91.
92.	    # 创建 DataLoader 对象，用于批量加载数据
93.	    data_iter = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
94.
95.	    return data_iter  # 假设图像是 28x28 的灰度图像
96.	#SGD优化函数
97.	def train_sgd(lr, batch_size, num_epochs):
98.	    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
99.	    return d2l.train_ch3(net, train_iter=train_iter, test_iter=test_iter, loss=loss, num_epochs=num_epochs, updater=optimizer)
100.	#调整学习率及批次
101.	gd_res = train_sgd(0.001, 1500, 10)
102.	sgd_res = train_sgd(0.005, 1,10)
103.	mini1_res = train_sgd(.4, 100,10)
104.	mini2_res = train_sgd(.05, 10,10)
105.	#固定学习率
106.	lr, num_epochs = 0.3, 10
107.	trainer = torch.optim.SGD(net.parameters(), lr=lr)
108.	device='cpu'
109.	train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
110.	#指数衰减
111.	class SquareRootScheduler:
112.	    def __init__(self, lr=0.1):
113.	        self.lr = lr
114.
115.	    def __call__(self, num_update):
116.	        return self.lr * pow(num_update + 1.0, -0.5)
117.	scheduler = SquareRootScheduler(lr=0.1)
118.	d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
119.	trainer = torch.optim.SGD(net.parameters(), lr)
120.	device='cpu'
121.	train(net, train_iter, test_iter, num_epochs, loss, trainer, device,scheduler)
122.	#多项式衰减
123.	trainer = torch.optim.SGD(net.parameters(), lr=0.5)
124.	scheduler2 = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
125.
126.	def get_lr(trainer, scheduler):
127.	    lr = scheduler2.get_last_lr()[0]
128.	    trainer.step()
129.	    scheduler2.step()
130.	    return lr
131.	d2l.plot(torch.arange(num_epochs), [get_lr(trainer, scheduler2)
132.	                                  for t in range(num_epochs)])
133.	train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
134.	    scheduler2)
135.	#余弦衰减
136.	class CosineScheduler:
137.	    def __init__(self, max_update, base_lr=0.01, final_lr=0,
138.	                warmup_steps=0, warmup_begin_lr=0):
139.	        self.base_lr_orig = base_lr
140.	        self.max_update = max_update
141.	        self.final_lr = final_lr
142.	        self.warmup_steps = warmup_steps
143.	        self.warmup_begin_lr = warmup_begin_lr
144.	        self.max_steps = self.max_update - self.warmup_steps
145.
146.	    def get_warmup_lr(self, epoch):
147.	        increase = (self.base_lr_orig - self.warmup_begin_lr) \
148.	                      * float(epoch) / float(self.warmup_steps)
149.	        return self.warmup_begin_lr + increase
150.
151.	    def __call__(self, epoch):
152.	        if epoch < self.warmup_steps:
153.	            return self.get_warmup_lr(epoch)
154.	        if epoch <= self.max_update:
155.	            self.base_lr = self.final_lr + (
156.	                self.base_lr_orig - self.final_lr) * (1 + math.cos(
157.	                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
158.	        return self.base_lr
159.
160.	scheduler3 = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
161.	d2l.plot(torch.arange(num_epochs), [scheduler3(t) for t in range(num_epochs)])
162.	trainer = torch.optim.SGD(net.parameters(), lr=0.3)
163.	train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
164.	      scheduler3)
165.	#预热
166.	scheduler4 = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
167.	d2l.plot(torch.arange(num_epochs), [scheduler4(t) for t in range(num_epochs)])
168.	trainer = torch.optim.SGD(net.parameters(), lr=0.3)
169.	train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
170.	      scheduler4)  