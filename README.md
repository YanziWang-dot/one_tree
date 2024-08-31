# SXXWAZ
苏小心和王阿紫一起快乐的学习python

<img src="https://github.com/YanziWang-dot/SXXWAZ/assets/101793579/a3f29363-51f1-469f-8059-32662afa5da9" width="400">


## [🎄苏小心点这里哦~  i love you](https://codepen.io/wangyanzi/embed/qBvBXGy?height=265&theme-id=dark&default-tab=result)

一个免费在线学习统计学知识的网站 [https://www.jmp.com/](https://www.jmp.com/zh_cn/statistics-knowledge-portal/exploratory-data-analysis/histogram.html)


</head>
<body>
<div class="tree">
  <div class="ornament">❤️</div>
  <div class="ornament">❄️</div>
  <div class="ornament">🌟</div>
  <div class="ornament">🎁</div>
</div>
</body>
</html>



## 苏小心的小笔记
## 王阿紫的小笔记




## 静态图：写的程序会创建一个计算的流程，然后再运行（先搭建好计算的框架，再统一运行）
## 动态图：随时做变量的定义，随时做计算和得到结果
开源的代码和开源的社区

torch.tensor([1,2,3], dtype=int) 其中dtype定义i数据类型
tensor = torch.tensor([[1,2,3],[4,5,6]])
tensor.ndim：查看维度 
tensor.shape，tensor.size()查看数据形状

数据生成
torch.ones(2,3) 两行三列的全1矩阵
torch.randint(0,10,(2,3))
a = torch.randn(3,4) 符合正态分布的
b = torch.rand_like(a,dtype=float) 生成一个和a很像的b,数据类型是float
b.view(6) 修改数据形状，相当于reshape
b[1]是一个tensor，把这个tensor变成一个数值，d[1].item() 
item()只能修改一个值，所以不能d.item(),只能d[i].item()
np.array  torch.tensor 数组和张量之间的相互转化

基本运算操作：
a + b对应位置相加 a.add_(b) ## 注意，任何使张量tensor发生变化的操作都有一个前缀
torch.add(a,b,out=result) out是输出结果在哪里
+ - * / % //都是对应位置加减乘除、取余、取整
矩阵乘法 torch.matmul() # 数据类型要匹配
转置.T
torch.sum(sample) 对sample tensor求和
torch.min(sample) torch.max(sample) torch.argmin(sample) 求最小值所在的位置
torch.mean(sample) 均值 torch.median(sample)

数据索引
索引从0开始，取数据时最后一个位置是不算的
for t in tensor:循环tensor中的每一个值

自动求导
BP算法
x = torch.ones((2,2),requires_grad = True)  其中，requires_grad = True的意思是可以进行梯度
out.backward() 求导计算梯度的值
print(x.grad) out对x求导的导数是？x的梯度是？

线性回归
reshape(-1,1) -1是自动匹配，列数为 1
构建模型一般把网络中具有可学习参数的层放在__init__()中
class LinearRegression(nn.Module):
  # 定义网络结构
  def __init__(self): 
      super(LinearRegression,self).__init__()  # 父类的初始化
      self.fc = nn.Linear(1,1)

  # 定义网络计算
  def forward(self,x):
      out = self.fc(x)
      return out

# 定义模型 
model = LinearRegression()
mse_loss = nn.MSEloss()
optimizer = optim.SGD(model.parameters(),lr = 0.1) 传入模型参数

# 查看模型参数
for name, parameters in model.named_parameters():
  print('name:{},param:{}'.format(name,paramters))

# 模型训练
for i in range(100):
  out = model(inputs)
  # 计算loss
  loss = mse_loss(out,target)
  # 梯度清0
  optimizer.zero_grad()
  










