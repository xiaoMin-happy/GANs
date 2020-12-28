# DCGAN

1. 将pooling层用Conv层代替

   对于判别模型：容许网络学习自己的空间下采样

   对于生成模型：容许网络学习自己的空间上采样

2. 在G和D上都使用batchnorm：

   - 解决初始化差的问题
   - 帮助梯度传播到每一层
   - 防止G 把所有的样本都收敛到同一个点

3. 在CNN中移除全连接层

4. 在G中除了输出层外的所有层均使用ReLU，输出层采用tanh

5. 在D的所有层上使用LeakyReLU



训练注意：

G：噪音向量先扩张(投影)然后reshape

边训练边测试，训练1次d训练两次g，效果会更好

BN：经验在卷积后激活函数好前效果好



训练出现的问题：

- 报错一：

```python
absl.flags._exceptions.IllegalFlagValueError: flag --train_size=inf: Expect argument to be a string or int, found <class 'float'>
```

原因：

类型错误：期望参数为string或int，发现<类“float”>

解决：

```
flags.DEFINE_integer  其中integer更改为float
```

- 报错二：

```python
import scipy.misc
scipy.misc.imsave()
AttributeError: module 'scipy.misc' has no attribute 'imread'
```

原因1：scipy版本过高

解决：降低scipy版本，pip install scipy==1.2.1

原因2：查看scipy.misc帮助文件得知，imread依赖于pillow

解决：在该python环境中 pip install Pillow



如何在终端运行程序：

- 进入python环境

```
- source activate
- conda activate env_name
```

- 转到工程所在文件夹

```
- cd 工程文件夹的路径
```

- 输入配置代码

```
python main.py --dataset faces_ --is_train True --is_crop True  （）
```

