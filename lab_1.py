import numpy as np
import cv2
from matplotlib import pyplot as plt


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def thresholding(img, t):
    res = np.where(img >= t, 255, 0)
    return res


def img_gist(img, img2, t1, t2, t3, t4):
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 8)
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title(t1)
    axs[0, 1].hist(img.flatten(), 255, [0, 255], color='blue')
    axs[0, 1].set_title(t2)
    axs[1, 0].imshow(img2, cmap='gray')
    axs[1, 0].set_title(t3)
    axs[1, 1].hist(img2.flatten(), 255, [0, 255], color='blue')
    axs[1, 1].set_title(t4)
    plt.show()


def distribution_func(img, img2):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(10, 8)
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    axs[0].plot(cdf_normalized)
    axs[0].set_title("Функция распределения до эквализации")
    hist2, bins2 = np.histogram(img2.flatten(), bins=256, range=[0, 256])
    cdf2 = hist2.cumsum()
    cdf_normalized2 = cdf2 / cdf2.max()
    axs[1].plot(cdf_normalized2)
    axs[1].set_title("Функция распределения после эквализации")
    plt.show()


def task1(img):
    img2 = thresholding(img, 127)
    img_gist(img, img2, 'Исходное изображение', 'Исходная гистограмма', 'Изображение после пороговой обработки',
             'Гистограмма после пороговой обработки')
    plt.plot(np.sort(img.flatten()), np.sort(img2.flatten()))
    plt.title('График функции поэлементного преобразования')
    plt.show()


def linear_func(x, a, b):
    x = a * x + b
    if x < 0:
        x = 0
    if x > 255:
        x = 255
    return x


def calc_param(img):
    f_min = np.min(img)
    f_max = np.max(img)
    g_min, g_max = 0, 255
    a = (g_max - g_min) / (f_max - f_min)
    b = (g_min * f_max - g_max * f_min) / (f_max - f_min)
    return a, b


def linear_transform(img):
    vec_transform = np.vectorize(linear_func)
    transform_img = vec_transform(img, calc_param(img)[0], calc_param(img)[1])
    return transform_img


def task_2(img):
    img2 = linear_transform(img)
    img_gist(img, img2, 'Исходное изображение', 'Исходная гистограмма', 'Изображение после линейного контрастирования',
             'Гистограмма после линейного контрастирования')
    x = np.linspace(0, 255, 256)
    y = calc_param(img)[0] * x + calc_param(img)[1]
    y[y < 0] = 0
    y[y > 255] = 255
    plt.scatter(x, y)
    plt.title('График функции линейного контрастирования')
    plt.show()


def equalization(img):
    g_min, g_max = 0, 255
    hist, bins = np.histogram(img.flatten(), 255, [0, 255])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    equal_img = (g_max - g_min) * cdf_normalized[img] + g_min
    equ = cv2.equalizeHist(img)
    return equal_img, equ


def task_3(img):
    img2, img3 = equalization(img)
    img_gist(img, img2, 'Исходное изображение', 'Исходная гистограмма', 'Изображение после эквализации',
             'Гистограмма после эквализации')
    img_gist(img2, img3, 'Изображение после самописной эквализации', 'Гистограмма после самописной эквализации',
             'Изображение после стандартной эквализации', 'Гистограмма после стандартной эквализации')
    distribution_func(img, img2)
    plt.plot(np.sort(img.flatten()), np.sort(img2.flatten()))
    plt.title('График функции поэлементного преобразования')
    plt.show()


def main():
    # path = '09_lena2.tif'
    path = '01_apc.tif'
    img = read_img(path)
    task1(img)
    task_2(img)
    task_3(img)


if __name__ == '__main__':
    main()
