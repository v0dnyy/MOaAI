import numpy as np
import cv2
from matplotlib import pyplot as plt
from lab_1 import linear_transform


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def prediction(m, n, y, predict_num):
    if m == 0 and n == 0:
        return 0
    else:
        if predict_num == 1:
            if n == 0:
                return y[m - 1][-1]
            return y[m][n - 1]
        if predict_num == 2:
            return int((y[m][n - 1] + y[m - 1][n]) / 2)


def MyDifCode(x, e, r):
    q = np.zeros(x.shape)
    y = np.zeros(x.shape)
    f = np.zeros(x.shape)
    for m in range(x.shape[0]):
        for n in range(x.shape[1]):
            p = prediction(m, n, y, r)
            f[m][n] = x[m][n] - p
            q[m][n] = np.sign(f[m][n]) * ((abs(f[m][n]) + e) // (2 * e + 1))
            y[m][n] = p + q[m][n] * (2 * e + 1)
    return q, f


def MyDifDeCode(q, e, r):
    y = np.zeros(q.shape)
    for m in range(q.shape[0]):
        for n in range(q.shape[1]):
            p = prediction(m, n, y, r)
            y[m][n] = p + q[m][n] * (2 * e + 1)
    return y


def calc_entropy(x):
    probably = np.unique(x, return_counts=True)[1]
    probably = probably / probably.sum()
    return -(probably * np.log2(probably)).sum()


def task_3_4(img):
    y_1 = [calc_entropy(MyDifCode(img, e, 1)[0]) for e in range(0, 50, 1)]
    y_2 = [calc_entropy(MyDifCode(img, e, 2)[0]) for e in range(0, 50, 1)]
    x = np.linspace(0, 51)
    plt.plot(x, y_1, c='r', label="предсказатель №1")
    plt.plot(x, y_2, c='g', label="предсказатель №2")
    plt.legend()
    plt.show()


def task_5(img):
    e = [5, 10, 20, 40]
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.flatten()
    for i in range(len(e)):
        q = MyDifCode(img, e[i], 1)[0]
        y = MyDifDeCode(q, e[i], 1)
        max_deviation = np.max(np.abs(y - img))
        if max_deviation <= e[i]:
            print(f"Контроль максимальной ошибки выполняется. Макс. отклонение = {max_deviation}")
        else:
            print(f"Контроль максимальной ошибки не выполняется. Макс. отклонение = {max_deviation}")
        axs[i].imshow(y, cmap="gray", vmin=0, vmax=255)
        axs[i].set_title(f"Восстановление e={e[i]}")
    plt.show()


def task_6_7(img):
    q_1_1, f_1 = linear_transform(MyDifCode(img, 0, 1)[0]), linear_transform(MyDifCode(img, 0, 1)[1])
    q_1_2, f_2 = linear_transform(MyDifCode(img, 0, 2)[0]), linear_transform(MyDifCode(img, 0, 2)[1])
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(f_1, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Разностный сигнал для предсказателя №1")
    axs[1].imshow(f_2, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title("Разностный сигнал для предсказателя №2")
    plt.show()
    q_2_1 = linear_transform(MyDifCode(img, 5, 1)[0])
    q_2_2 = linear_transform(MyDifCode(img, 5, 2)[0])
    q_3_1 = linear_transform(MyDifCode(img, 10, 1)[0])
    q_3_2 = linear_transform(MyDifCode(img, 10, 2)[0])
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    fig.suptitle('Квантованный разностный сигнал для предсказателя №1 при')
    axs[0].imshow(q_1_1, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("e = 0")
    axs[1].imshow(q_2_1, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title("e = 5")
    axs[2].imshow(q_3_1, cmap='gray', vmin=0, vmax=255)
    axs[2].set_title("e = 10")
    plt.show()
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    fig.suptitle('Квантованный разностный сигнал для предсказателя №2 при')
    axs[0].imshow(q_1_2, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("e = 0")
    axs[1].imshow(q_2_2, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title("e = 5")
    axs[2].imshow(q_3_2, cmap='gray', vmin=0, vmax=255)
    axs[2].set_title("e = 10")
    plt.show()


def main():
    # path = '09_lena2.tif'
    # path = '01_apc.tif'
    path = '11_peppers2.tif'
    img = read_img(path)

    #task_3_4(img)
    task_5(img)
    task_6_7(img)
    #Готово



if __name__ == '__main__':
    main()
