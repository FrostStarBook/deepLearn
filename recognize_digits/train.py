from __future__ import print_function  # 将python3中的print特性导入当前版本

import os
import argparse
from PIL import Image  # 导入图像处理模块
import numpy
import paddle  # 导入paddle模块
import paddle.fluid as fluid


# 启动参数,默认使用CPU
def parse_args():
    parser = argparse.ArgumentParser("mnist")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=5, help="number of epochs.")
    args = parser.parse_args()
    return args


def loss_net(hidden, label):
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    # 使用类交叉熵函数计算predict和label之间的损失函数
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    # 计算平均损失
    avg_loss = fluid.layers.mean(loss)
    # 计算分类准确率
    acc = fluid.layers.accuracy(input=prediction, label=label)
    # 返回计算后的数据
    return prediction, avg_loss, acc


# 多层感知器
# 下面代码实现了一个含有两个隐藏层（即全连接层）的多层感知器。其中两个隐藏层的激活函数均采用tanh，输出层的激活函数用Softmax
def multilayer_perceptron(img, label):
    # 第一个全连接层，激活函数为tanh
    img = fluid.layers.fc(input=img, size=200, act='tanh')
    # 第二个全连接层，激活函数为tanh
    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    return loss_net(hidden, label)


# Softmax回归
# 只通过一层简单的以softmax为激活函数的全连接层，就可以得到分类的结果
def softmax_regression(img, label):
    return loss_net(img, label)


# 卷积神经网络LeNet-5
# 输入的二维图像，首先经过两次卷积层到池化层，再经过全连接层，最后使用以softmax为激活函数的全连接层作为输出层
def convolutional_neural_network(img, label):
    # 第一个卷积-池化层
    # 使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)

    # 第二个卷积-池化层
    # 使用50个5*5的滤波器，池化大小为2，池化步长为2，激活函数为Relu
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


# 主训练函数
def train(nn_type,
          use_cuda,
          save_dirname=None,
          model_filename=None,
          params_filename=None):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.default_startup_program()  # 设置 start program
    main_program = fluid.default_main_program()  # 设置main program

    if args.enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
        startup_program.random_seed = 90
        main_program.random_seed = 90
    else:
        # 每次读取训练集中的500个数据并随机打乱，传入batched reader中，batched reader 每次 yield 64个数据
        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
            batch_size=BATCH_SIZE)
        # 读取测试集的数据，每次 yield 64个数据
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

    # 输入的原始图像数据，大小为28*28*1
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 标签层，名称为label,对应输入图片的类别标签
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if nn_type == 'softmax_regression':
        net_conf = softmax_regression
    elif nn_type == 'multilayer_perceptron':
        net_conf = multilayer_perceptron
    else:
        net_conf = convolutional_neural_network

    # 调用train_program 获取预测值，损失值，
    prediction, avg_loss, acc = net_conf(img, label)

    # 克隆一份test program
    test_program = main_program.clone(for_test=True)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    # 训练测试集
    def train_test(train_test_program, train_test_feed, train_test_reader):
        # 将分类准确率存储在acc_set中
        acc_set = []
        # 将平均损失存储在avg_loss_set中
        avg_loss_set = []
        # 将测试 reader yield 出的每一个数据传入网络中进行训练
        for test_data in train_test_reader():
            acc_np, avg_loss_np = exe.run(
                program=train_test_program,
                feed=train_test_feed.feed(test_data),
                fetch_list=[acc, avg_loss])
            acc_set.append(float(acc_np))
            avg_loss_set.append(float(avg_loss_np))
        # 获得测试数据上的准确率和损失值
        acc_val_mean = numpy.array(acc_set).mean()
        avg_loss_val_mean = numpy.array(avg_loss_set).mean()

        # 返回平均损失值，平均准确率
        return avg_loss_val_mean, acc_val_mean

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    # 创建执行器
    exe = fluid.Executor(place)

    # 输入的原始图像数据，名称为img，大小为28*28*1
    # 标签层，名称为label,对应输入图片的类别标签
    # 告知网络传入的数据分为两部分，第一部分是img值，第二部分是label值
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    exe.run(startup_program)
    epochs = [epoch_id for epoch_id in range(PASS_NUM)]  # 设置训练过程的超参

    lists = []
    step = 0
    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(
                main_program,
                feed=feeder.feed(data),
                fetch_list=[avg_loss, acc])
            if step % 100 == 0:  # #每训练100次 打印一次log
                # 打印训练的中间结果，训练轮次，batch数，损失函数
                print("Pass %d, Batch %d, cost %f" % (step, epoch_id,
                                                      metrics[0]))
            step += 1
        # 测试每个epoch的分类效果
        avg_loss_val, acc_val = train_test(
            train_test_program=test_program,
            train_test_reader=test_reader,
            train_test_feed=feeder)

        print("测试阶段 %d, avg_cost: %s, acc: %s" %
              (epoch_id, avg_loss_val, acc_val))
        lists.append((epoch_id, avg_loss_val, acc_val))
        # 保存训练好的模型参数用于预测
        if save_dirname is not None:
            fluid.io.save_inference_model(
                save_dirname, ["img"], [prediction],
                exe,
                model_filename=model_filename,
                params_filename=params_filename)

    if args.enable_ce:
        print("kpis\ttrain_cost\t%f" % metrics[0])
        print("kpis\ttest_cost\t%s" % avg_loss_val)
        print("kpis\ttest_acc\t%s" % acc_val)

    # 选择效果最好的pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print('最好的模型为: %s, 测试平局损失为: %s' % (best[0], best[1]))
    print('分类精度为: %.2f%%' % (float(best[2]) * 100))


# 预测
def infer(use_cuda,
          save_dirname=None,
          model_filename=None,
          params_filename=None):
    if save_dirname is None:
        return

    # 设置在CPU处理
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # 加载图片
    def load_image(file):
        # 读取图片文件，并将它转成灰度图
        im = Image.open(file).convert('L')
        # 将输入图片调整为 28*28 的高质量图
        im = im.resize((28, 28), Image.ANTIALIAS)
        # 将图片转换为numpy
        im = numpy.array(im).reshape(1, 1, 28, 28).astype(numpy.float32)
        # 对数据作归一化处理
        im = im / 255.0 * 2.0 - 1.0
        return im

    # 构建图片path
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    tensor_img = load_image(cur_dir + '/image/infer_3.png')

    """
        Inference 创建及预测
    """
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # 使用 fluid.io.load_inference_model 获取 inference program desc,
        # feed_target_names 用于指定需要传入网络的变量名
        # fetch_targets 指定希望从网络中fetch出的变量名
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
            save_dirname, exe, model_filename, params_filename)

        # 将feed构建成字典 {feed_target_name: feed_target_data}
        # 结果将包含一个与fetch_targets对应的数据列表
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_img},
            fetch_list=fetch_targets)
        lab = numpy.argsort(results)

        # 打印 infer_3.png 这张图片的预测结果
        print("推测结果 image/infer_3.png 是: %d" % lab[0][0][-1])


def main(use_cuda, nn_type):
    model_filename = None
    params_filename = None
    save_dirname = "recognize_digits_" + nn_type + ".model"

    # 使用本地参数调用train()来运行分布式训练
    train(
        nn_type=nn_type,
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename)
    infer(
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename)


if __name__ == '__main__':
    args = parse_args()
    BATCH_SIZE = 64  # 一个minibatch中有64个数据
    PASS_NUM = args.num_epochs  # 设置训练轮数
    use_cuda = args.use_gpu  # 默认使用CPU训练
    # predict = 'softmax_regression' # 取消注释将使用 Softmax回归
    # predict = 'multilayer_perceptron' # 取消注释将使用 多层感知器
    predict = 'convolutional_neural_network'  # LeNet5卷积神经网络
    main(use_cuda=use_cuda, nn_type=predict)
