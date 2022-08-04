# 2.2节，蒙特卡洛近似计算圆周率。
import numpy as np


def approxiate_pi(n: int):
    # 在[-1, 1] x [-1, 1]的空间中随机取n个点。
    x_lst = np.random.uniform(-1, 1, size=n)
    y_lst = np.random.uniform(-1, 1, size=n)
    # 统计距离圆心距离在1以内的点。
    m = 0
    for x, y in zip(x_lst, y_lst):
        if x ** 2 + y ** 2 <= 1:
            m += 1
    # 近似计算圆周率。
    pi = 4 * m / n
    return pi


if __name__ == "__main__":
    pi = approxiate_pi(100)
    print("100个点近似的圆周率：", pi)

    pi = approxiate_pi(10000)
    print("10000个点近似的圆周率：", pi)

    pi = approxiate_pi(1000000)
    print("1000000个点近似的圆周率：", pi)
