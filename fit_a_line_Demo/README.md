# 线性回归

### 背景介绍
给定一个大小为n的数据集 yi,xi1,...,xidni=1，其中xi1,…,xid是第i个样本d个属性上的取值，yi是该样本待预测的目标。线性回归模型假设目标yi可以被属性间的线性组合描述，即

![image](https://github.com/liushuheng163/deepLearn/blob/master/fit_a_line_Demo/image/formula_fit_a_line_1.png?raw=true)

例如，在我们将要建模的房价预测问题里，xij是描述房子i的各种属性（比如房间的个数、周围学校和医院的个数、交通状况等），而 yi是房屋的价格。

初看起来，这个假设实在过于简单了，变量间的真实关系很难是线性的。但由于线性回归模型有形式简单和易于建模分析的优点，它在实际问题中得到了大量的应用。很多经典的统计学习、机器学习书籍也选择对线性模型独立成章重点讲解。

### 效果展示
我们使用从UCI Housing Data Set获得的波士顿房价数据集进行模型的训练和预测。效果如下
![image](https://github.com/liushuheng163/deepLearn/blob/master/fit_a_line_Demo/image/%E8%AE%AD%E7%BB%83%E7%BB%93%E6%9E%9C.png?raw=true)

### 模型定义
在波士顿房价数据集中，和房屋相关的值共有14个：前13个用来描述房屋相关的各种信息，即模型中的 xi；最后一个值为我们要预测的该类房屋价格的中位数，即模型中的 yi。因此，我们的模型就可以表示成：
![image](https://github.com/liushuheng163/deepLearn/blob/master/fit_a_line_Demo/image/formula_fit_a_line_2.png?raw=true)

Ŷ  表示模型的预测结果，用来和真实值Y区分。模型要学习的参数即：ω1,…,ω13,b。
建立模型后，我们需要给模型一个优化目标，使得学到的参数能够让预测值Ŷ 尽可能地接近真实值Y。这里我们引入损失函数（Loss Function，或Cost Function）这个概念。 输入任意一个数据样本的目标值yi和模型给出的预测值yi^，损失函数输出一个非负的实值。这个实值通常用来反映模型误差的大小。

对于线性回归模型来讲，最常见的损失函数就是均方误差（Mean Squared Error， MSE）了，它的形式是：

![image](https://github.com/liushuheng163/deepLearn/blob/master/fit_a_line_Demo/image/formula_fit_a_line_3.png?raw=true)



即对于一个大小为n的测试集，MSE是n个数据预测结果误差平方的均值。

对损失函数进行优化所采用的方法一般为梯度下降法。梯度下降法是一种一阶最优化算法。如果f(x)在点xn有定义且可微，则认为f(x)在点xn沿着梯度的负方向−▽f(xn)下降的是最快的。反复调节x，使得f(x)接近最小值或者极小值，调节的方式为：

![image](https://github.com/liushuheng163/deepLearn/blob/master/fit_a_line_Demo/image/formula_fit_a_line_4.png?raw=true)


其中λ代表学习率。这种调节的方法称为梯度下降法。

### 环境
- python3.7.5
- macos 10.15.1

### 基本包安装
#### 安装CPU版本的PaddlePaddle 
执行以下命令安装（使用清华源）：
```
python3 -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 验证
进入Python3解释器,输入
```python
import paddle.fluid
paddle.fluid.install_check.run_check()
```
如果出现
```
Your Paddle Fluid is installed succesfully!
```
则安装成功



