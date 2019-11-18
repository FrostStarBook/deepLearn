from __future__ import print_function

import sys
import argparse

import math
import numpy

import paddle
import paddle.fluid as fluid


# 启动参数
def parse_args():
    parser = argparse.ArgumentParser("fit_a_line")
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
        '--num_epochs', type=int, default=100, help="number of epochs.")
    args = parser.parse_args()
    return args


# 训练测试损失
def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
    return [x_d / count for x_d in accumulated]


# 保存图片
def save_result(points1, points2):
    import matplotlib
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties

    # matplotlib 设置中文字体
    matplotlib.use('Agg')
    rcParams['axes.unicode_minus'] = False
    chinese_font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc',
                                  size=15)
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2

    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('预测值 VS 真实值 ', fontproperties=chinese_font)
    plt.ylabel(u'房价', fontproperties=chinese_font)
    plt.legend((u'预测值', u'真实值'), loc='best', prop=chinese_font)
    plt.savefig('./image/训练结果.png', fontproperties=chinese_font)


def main():
    """
        定义用于训练的数据提供器
        提供器每次读入一个大小为batch_size的数据批次
    """
    batch_size = 10

    # paddle自带数据包 uci_housing
    if args.enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=batch_size)
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.uci_housing.train(), buf_size=500),
            batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.uci_housing.test(), buf_size=500),
            batch_size=batch_size)

    """
        配置训练程序
    """
    # 特征向量的长度为13
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')  # 定义输入的形状和数据类型
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')  # 定义输出的形状和数据类型

    main_program = fluid.default_main_program()  # 获取默认/全局主函数
    startup_program = fluid.default_startup_program()  # 获取默认/全局启动程序

    if args.enable_ce:
        main_program.random_seed = 90
        startup_program.random_seed = 90

    y_predict = fluid.layers.fc(input=x, size=1, act=None)  # 连接输入和输出的全连接层
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)  # 利用标签数据和输出的预测数据估计方差
    avg_loss = fluid.layers.mean(cost)  # 对方差求均值，得到平均损失

    """
        Optimizer Function 优化器配置
    """
    # 克隆main_program得到test_program
    # 有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
    # 该api不会删除任何操作符,请在backward和optimization之前使用
    test_program = main_program.clone(for_test=True)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)  # 学习率
    sgd_optimizer.minimize(avg_loss)

    """
        定义运算场所
        指定使用CPU运算
    """
    # 定义运算是发生在CPU还是GPU
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()  # 指明executor的执行场所
    # executor可以接受传入的program，并根据feed map(输入映射表)和fetch list(结果获取表)向program中添加数据输入算子和结果获取算子。使用close()关闭该executor，
    # 调用run(...)执行program。
    exe = fluid.Executor(place)

    # 指定保存参数的目录
    params_dirname = "fit_a_line.model"
    num_epochs = args.num_epochs

    """
        训练主循环
    """
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe.run(startup_program)

    train_prompt = "训练损失"
    test_prompt = "测试损失"
    step = 0

    exe_test = fluid.Executor(place)

    # 设置训练主循环
    for pass_id in range(num_epochs):
        for data_train in train_reader():
            avg_loss_value, = exe.run(
                main_program,
                feed=feeder.feed(data_train),
                fetch_list=[avg_loss])
            if step % 10 == 0:  # 每10个批次记录并输出一下训练损失log
                print("%s, 批次 %d, 损失值 %f" %
                      (train_prompt, step, avg_loss_value[0]))

            if step % 100 == 0:  # 每100批次记录并输出一下测试损失log
                test_metics = train_test(
                    executor=exe_test,
                    program=test_program,
                    reader=test_reader,
                    fetch_list=[avg_loss],
                    feeder=feeder)
                print("%s, 批次 %d, 损失值 %f" %
                      (test_prompt, step, test_metics[0]))
                # 如果准确度满足要求,结束训练
                if test_metics[0] < 10:
                    break

            step += 1

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NaN loss, 训练失败!!!")
        if params_dirname is not None:
            # 训练成功,保存训练参数到params_dirname
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict],
                                          exe)

        if args.enable_ce and pass_id == args.num_epochs - 1:
            print("kpis\ttrain_cost\t%f" % avg_loss_value[0])
            print("kpis\ttest_cost\t%f" % test_metics[0])

    """
        预测
        需要构建一个使用训练好的参数来进行预测的程序，训练好的参数位置在params_dirname
    """
    # 准备预测环境    类似于训练过程，预测器需要一个预测程序来做预测
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    # 通过fluid.io.load_inference_model，预测器会从params_dirname中读取已经训练好的模型，来对从未遇见过的数据进行预测
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets
         ] = fluid.io.load_inference_model(params_dirname, infer_exe)  # 载入预训练模型
        batch_size = 10

        infer_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=batch_size)  # 准备测试集

        infer_data = next(infer_reader())
        infer_feat = numpy.array(
            [data[0] for data in infer_data]).astype("float32")  # 提取测试集中的数据
        infer_label = numpy.array(
            [data[1] for data in infer_data]).astype("float32")  # 提取测试集中的标签

        assert feed_target_names[0] == 'x'

        # 进行预测
        results = infer_exe.run(
            inference_program,
            feed={feed_target_names[0]: numpy.array(infer_feat)},
            fetch_list=fetch_targets)

        print("预测结果: (房价)")  # 打印预测结果和标签并可视化结果
        for idx, val in enumerate(results[0]):
            print("%d: %.2f" % (idx, val))  # 打印预测结果

        print("\n真实值:")
        for idx, val in enumerate(infer_label):
            print("%d: %.2f" % (idx, val))  # 打印真实值

        # 保存图片
        # 由于每次都是随机选择一个minibatch的数据作为当前迭代的训练数据，所以每次得到的预测结果会有所不同。
        save_result(results[0], infer_label)


if __name__ == '__main__':
    args = parse_args()
    main()
