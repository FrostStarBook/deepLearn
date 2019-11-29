from __future__ import print_function

import os
import argparse
import paddle
import paddle.fluid as fluid
import numpy
import sys
from vgg import vgg_bn_drop
from resnet import resnet_cifar10


def parse_args():
    parser = argparse.ArgumentParser("image_classification")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='If set, run the task with continuous evaluation logs.')
    parser.add_argument(
        '--use_gpu', type=bool, default=0, help='whether to use gpu')
    parser.add_argument(
        '--num_epochs', type=int, default=1, help='number of epoch')
    args = parser.parse_args()
    return args


def inference_network():
    # 图像是32 * 32的RGB表示.
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')

    predict = resnet_cifar10(images, 32)
    # predict = vgg_bn_drop(images) # 取消注释使用 vgg net
    return predict


def train_network(predict):
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


def train(use_cuda, params_dirname):
    # 设置CPU训练
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    BATCH_SIZE = 128

    # 指定加载数据
    if args.enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)
    else:
        test_reader = paddle.batch(
            paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.cifar.train10(), buf_size=128 * 100),
            batch_size=BATCH_SIZE)

    feed_order = ['pixel', 'label']

    main_program = fluid.default_main_program()
    start_program = fluid.default_startup_program()

    if args.enable_ce:
        main_program.random_seed = 90
        start_program.random_seed = 90

    predict = inference_network()
    avg_cost, acc = train_network(predict)

    # clone一份测试程序
    test_program = main_program.clone(for_test=True)
    optimizer = optimizer_program()
    optimizer.minimize(avg_cost)

    exe = fluid.Executor(place)

    EPOCH_NUM = args.num_epochs

    # 训练测试损失
    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = len([avg_cost, acc]) * [0]
        for tid, test_data in enumerate(reader()):
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=[avg_cost, acc])
            accumulated = [
                x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
            ]
            count += 1
        return [x / count for x in accumulated]

    # 主训练循环
    def train_loop():
        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(start_program)

        step = 0
        for pass_id in range(EPOCH_NUM):
            for step_id, data_train in enumerate(train_reader()):
                avg_loss_value = exe.run(
                    main_program,
                    feed=feeder.feed(data_train),
                    fetch_list=[avg_cost, acc])
                if step_id % 100 == 0:
                    print("\n通过 %d, 批次 %d, 代价损失 %f, 准确率 %f" % (
                        step_id, pass_id, avg_loss_value[0], avg_loss_value[1]))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                step += 1

            avg_cost_test, accuracy_test = train_test(
                test_program, reader=test_reader)
            print('\n测试通过 {0}, 损耗 {1:2.2}, 准确率 {2:2.2}%'.format(
                pass_id, avg_cost_test, accuracy_test))

            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ["pixel"],
                                              [predict], exe)

            if args.enable_ce and pass_id == EPOCH_NUM - 1:
                print("kpis\ttrain_cost\t%f" % avg_loss_value[0])
                print("kpis\ttrain_acc\t%f" % avg_loss_value[1])
                print("kpis\ttest_cost\t%f" % avg_cost_test)
                print("kpis\ttest_acc\t%f" % accuracy_test)

    train_loop()


def infer(use_cuda, params_dirname=None):
    from PIL import Image
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    def load_image(infer_file):
        im = Image.open(infer_file)
        im = im.resize((32, 32), Image.ANTIALIAS)

        im = numpy.array(im).astype(numpy.float32)
        # 加载图像的存储顺序是W(宽度), H(高度)，C(通道) PaddlePaddle要求
        im = im.transpose((2, 0, 1))  # CHW
        im = im / 255.0

        # 添加一维以模仿列表格式
        im = numpy.expand_dims(im, axis=0)
        return im

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    image_path = '/image/automobile4.png'

    img = load_image(cur_dir + image_path)

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        # 将Feed构造为{feed_target_name：feed_target_data}的字典,结果将包含与fetch_targets对应的数据列表
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: img},
            fetch_list=fetch_targets)

        # 预测列表
        label_list = [
            "飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙",
            "马", "船", "卡车"
        ]

        print("预测图片  " + image_path + "    预测结果: %s" % label_list[numpy.argmax(results[0])])


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    # 保存路径
    save_path = "image_classification_resnet.inference.model"

    # 训练函数
    # train(use_cuda=use_cuda, params_dirname=save_path)

    # 预测函数
    infer(use_cuda=use_cuda, params_dirname=save_path)


if __name__ == '__main__':
    # 默认使用CPU训练,有N卡的可以切换GPU训练
    args = parse_args()
    use_cuda = args.use_gpu
    main(use_cuda)
