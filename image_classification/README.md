# 图像分类


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



