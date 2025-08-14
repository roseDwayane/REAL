

def schedule_normal(epoch):
    lr = 0.001
    r = epoch // 20
    lr -= lr/10 * r
    lr = max(1.0e-4, lr)
    print('Learning-rate {: 0.5f}'.format(lr))
    return lr


def schedule_step(epoch):
    lr0 = 0.001
    lr1 = 0.0002
    lr = lr0 if epoch < 50 else lr1
    print('Learning-rate {: 0.5f}'.format(lr))
    return lr


def schedule_cos(epoch):
    import numpy as np
    import math

    def step(x):
        return 1.0 if x >= 0 else 0.0

    lr_base = 0.001

    t_linear_up = 10
    t_cos = 50

    epoch_r = epoch % (t_linear_up + t_cos)

    lr_linear_up = 1.0/t_linear_up*epoch_r * step(t_linear_up - epoch_r)
    lr_cos = step(epoch_r - t_linear_up) * (math.cos(2*np.pi*(epoch_r - t_linear_up)/(t_cos*2)) + 1.0)/2.0
    lr = max(lr_base * (lr_linear_up + lr_cos), 1.0e-4)
    lr = min(lr, lr_base)
    #print('Learning-rate cos {: 0.5f}'.format(lr))
    return lr


def schedule_cos0(epoch):
    import numpy as np
    import math

    def step(x):
        return 1.0 if x >= 0 else 0.0

    t_warm_up = 10
    t_cos = 40
    lr_base = 0.001

    lr_warm_up = 1.0/t_warm_up*epoch * step(t_warm_up - epoch)
    lr_cos = step(epoch - t_warm_up) * (math.cos(2*np.pi*(epoch-t_warm_up)/(t_cos*2)) + 1.0)/2.0
    lr = max(lr_base * (lr_warm_up + lr_cos), 1.0e-4)
    print('Learning-rate {: 0.5f}'.format(lr))
    return lr


def schedule_cosine_decay(epoch):
    import numpy as np
    import math

    def step(x):
        return 1.0 if x >= 0 else 0.0

    def decay(epoch_):
        duration = [int(30*np.float_power(1.5, n)) for n in range(10)]
        th = 0
        for n in range(10):
            tn = duration[n]
            if epoch_ > th + tn:
                th += tn
                continue

            epoch_r_ = epoch_ - th
            t_linear_up_ = 1.0/5.0 * tn
            t_cos_ = 4.0/5.0 * tn
            return epoch_r_, t_linear_up_, t_cos_

    lr_base = 0.001
    epoch_r, t_linear_up, t_cos = decay(epoch)

    lr_linear_up = 1.0/t_linear_up*epoch_r * step(t_linear_up - epoch_r)
    lr_cos = step(epoch_r - t_linear_up) * (math.cos(2*np.pi*(epoch_r - t_linear_up)/(t_cos*2)) + 1.0)/2.0
    lr = max(lr_base * (lr_linear_up + lr_cos), 1.0e-4)
    lr = min(lr, lr_base)
    #print('Learning-rate cos {: 0.5f}'.format(lr))
    return lr


def schedule_sawtooth(epoch):
    import math
    lr_t = 50
    lr_base = 0.001
    lr_sawtooth = (epoch % lr_t) / lr_t
    lr = lr_base*lr_sawtooth + 1.0e-5
    print('Learning rate', lr)
    return lr


def schedule_rand(epoch):
    import numpy as np
    lr_base = 0.001
    lr = lr_base*np.random.rand(1).item() + 1.0e-5
    print('Learning rate', lr)
    return lr


def cosine_decay_restarts(epoch):
    import tensorflow as tf
    lr_decayed = tf.train.cosine_decay_restarts(learning_rate=0.001, global_step=epoch, first_decay_steps=100)
    return lr_decayed


def schedule_step_decrease_closure(lr_init=0.001, interval=50):
    """
    利用闭包实现可配置的学习率调度. 每interval个回合, 衰减1/10
    :param lr_init:
    :param interval: decrease the learning rate by 10x per interval epochs
    :return:
    """
    # inner是内函数
    def inner(cur_epoch):
        # 在内函数中, 用到了外函数的临时变量
        r = cur_epoch // interval
        lr = lr_init / (10.0 ** r)
        print('Learning-rate {: 0.4e} @step {: }'.format(lr, r))
        return lr

    # 外函数的返回值是内函数的引用
    return inner


def schedule_none_closure(lr_init=0.001):
    """
    伪调度.
    :param lr_init:
    :return:
    """
    lr0 = lr_init

    def inner(epoch):
        lr = lr0
        print('None learning rate {: 0.5f}'.format(lr))
        return lr

    return inner


def schedule_normal_closure(lr_init=0.001, epochs=100, interval=10):
    """
    :param lr_init:
    :param epochs:
    :param interval: decay per interval epochs
    :return:
    """
    lr0 = lr_init
    interval = interval

    def inner(epoch):
        lr = lr0
        r = epoch // interval
        lr -= lr/(epochs//interval) * r
        lr = max(1.0e-5, lr)
        print('Learning-rate {: 0.5f}'.format(lr))
        return lr

    return inner


def schedule_step_closure(lr_init=0.001, epochs=100, interval=10):
    """
    :param lr_init:
    :param epochs:
    :param interval: decay per interval epochs
    :return:
    """
    lr0 = lr_init
    interval = interval
    n_steps = epochs // interval
    dlr = lr0/n_steps

    def inner(cur_epoch):
        lr = lr0
        r = cur_epoch // interval
        lr -= dlr * r
        lr = max(1.0e-6, lr)
        print('Learning-rate {: 0.5f}'.format(lr))
        return lr

    return inner


def schedule_cosine_closure(lr_base=0.001):
    """
    :param lr_base:
    :return:
    """
    import numpy as np
    import math

    lr_base = lr_base

    def inner(epoch):
        def u(x):
            return 1.0 if x >= 0 else 0.0

        t_warm_up = 10
        t_cos = 40

        lr_warm_up = 1.0/t_warm_up*epoch * u(t_warm_up - epoch)
        lr_cos = u(epoch - t_warm_up) * (math.cos(2*np.pi*(epoch-t_warm_up)/(t_cos*2)) + 1.0)/2.0
        lr = max(lr_base * (lr_warm_up + lr_cos), 1.0e-4)
        lr = min(lr, lr_base)
        #print('Learning-rate {: 0.4e}'.format(lr))
        return lr

    return inner


def schedule_cosine_decay_closure(lr_base=0.001):
    """
    :param lr_base:
    :return:
    """
    import numpy as np
    import math

    lr_base = lr_base

    def inner(epoch):
        def step(x):
            return 1.0 if x >= 0 else 0.0

        def decay(epoch_):
            duration = [int(30*np.float_power(1.5, n)) for n in range(10)]
            th = 0
            for n in range(10):
                tn = duration[n]
                if epoch_ > th + tn:
                    th += tn
                    continue

                epoch_r_ = epoch_ - th
                t_linear_up_ = 0.2 * tn
                t_cos_ = 0.8 * tn
                return epoch_r_, t_linear_up_, t_cos_

        epoch_r, t_linear_up, t_cos = decay(epoch)

        lr_linear_up = 1.0/t_linear_up*epoch_r * step(t_linear_up - epoch_r)
        lr_cos = step(epoch_r - t_linear_up) * (math.cos(2*np.pi*(epoch_r - t_linear_up)/(t_cos*2)) + 1.0)/2.0
        lr = max(lr_base * (lr_linear_up + lr_cos), 1.0e-4)
        lr = min(lr, lr_base)
        #print('Learning-rate cos {: 0.5f}'.format(lr))
        return lr
    return inner


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    f = schedule_step_closure(lr_init=0.001, epochs=200, interval=25)
    lr_list = []
    for epoch in range(200):
        lr_list.append(f(epoch))

    plt.plot(lr_list)
    plt.show()

"""
example:
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
"""




