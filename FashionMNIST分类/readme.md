从零开始实现那个太麻烦了,线性回归实现过一次了,就不想再实现一次了,因为差不多,就不想写了,进企业之后基本都是调用框架写的,所以这里就没有从零开始实现的代码了,只有调用pytorch

Config文件是配置文件,对里面很多参数进行了设置,如随机数种子,批大小,epoch数量,输入层大小,输出层大小.

util文件是一些函数,如画图,获取对应的label,以及读取数据.然后画图和获取label的两个函数,做了判断,李沐老师那个是只有批量才能运行,我进行了if判断,然后单个的也可以运行.

softmax回归实现图像分类文件是softmax主程序,调用pytorch框架实现FashionMNIST数据集,其中由于pycharm动态绘图好像不行的,所以我使用tensorboardX来实现训练可视化.然后精度,李沐老师是自己实现的,然后我嫌太麻烦了,就调用sklearn.metric的精度的包了.

多层感知机.py是多层感知机的实现,精度会比softmax高一点点.

多层感知机dropout,py是多层感知机加了dropout层.

My_LeNet.py是LeNet模型的实现.不知道是不是因为初始化的原因,我要第二个epoch开始才会收敛.然后用的是最大池化(讲道理,图像处理不应该都是用最大池化的吗?)效果比视频里要好, 貌似第九个epoch在测试集上效果最好,可能用Adam效果会更好一点.

My_AlexNet.py是AlexNet的实现,代码是可以在CPU和GPU上运行的,CPU上运行要花很长时间.由于作者没有GPU,使用了colab实现,

My_ResNet.py是ResNet的实现,同AlexNet.py.
