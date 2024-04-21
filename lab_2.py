import numpy as np
import cv2
from scipy.signal import convolve2d
from matplotlib import pyplot as plt


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def plot_task_1(img, s1, s2, gradient, outline_img):
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(15, 8)
    axs[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title("Исходное изображение")
    axs[0, 1].imshow(s1 + 128, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title("Частная производная по горизонтали")
    axs[0, 2].imshow(s2 + 128, cmap='gray', vmin=0, vmax=255)
    axs[0, 2].set_title("Частная производная по вертикали")
    axs[1, 0].imshow(gradient, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].set_title("Оценка модуля градиента функции яркости")
    axs[1, 1].hist(gradient.flatten(), bins=256, range=[0, 256])
    axs[1, 1].set_title("Гистограмма оценки модуля градиента")
    axs[1, 2].imshow(outline_img, cmap='gray', vmin=0, vmax=255)
    axs[1, 2].set_title("Полученные контуры")
    fig.suptitle('Метод простого градиента')
    plt.show()


def simple_gradient(img):
    horisont_window = np.array([[-1, 1]])
    vertical_window = np.array([[-1], [1]])
    horisontal_dif = convolve2d(img, horisont_window, mode='same', boundary='symm')  # свертка
    vertical_dif = convolve2d(img, vertical_window, mode='same', boundary='symm')
    gradient = np.sqrt(horisontal_dif ** 2 + vertical_dif ** 2)
    outline_img = (gradient >= 30) * 255
    plot_task_1(img, horisontal_dif, vertical_dif, gradient, outline_img)
    return outline_img


def plot_task_2(img, laplacian, outline_img):
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(15, 8)
    axs[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title("Исходное изображение")
    axs[0, 1].imshow(laplacian, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title("Оценка лапласиана")
    axs[1, 0].hist(laplacian.flatten(), bins=256, range=[0, 256])
    axs[1, 0].set_title("Гистограмма оценки лапласиана")
    axs[1, 1].imshow(outline_img, cmap='gray', vmin=0, vmax=255)
    axs[1, 1].set_title("Полученные контуры")
    fig.suptitle('Метод аппроксимации лапласиана')
    plt.show()


def laplacian_approx(img):
    window_1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian = convolve2d(img, window_1, mode='same', boundary='symm')
    outline_img = (laplacian >= 30) * 255
    plot_task_2(img, laplacian, outline_img)
    return outline_img


def plot_task_3(img, s1, s2, gradient, outline_img):
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(15, 8)
    axs[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title("Исходное изображение")
    axs[0, 1].imshow(s1 + 128, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title("S1")
    axs[0, 2].imshow(s2 + 128, cmap='gray', vmin=0, vmax=255)
    axs[0, 2].set_title("S2")
    axs[1, 0].imshow(gradient, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].set_title("Оператор Прюитт")
    axs[1, 1].hist(gradient.flatten(), bins=256, range=[0, 256])
    axs[1, 1].set_title("Гистограмма операторa Прюитт")
    axs[1, 2].imshow(outline_img, cmap='gray', vmin=0, vmax=255)
    axs[1, 2].set_title("Полученные контуры")
    fig.suptitle('Метод оператора Прюитт')
    plt.show()


def pruitt(img):
    horisont_window = 1 / 6 * np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    vertical_window = 1 / 6 * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    s1 = convolve2d(img, horisont_window, mode='same', boundary='symm')
    s2 = convolve2d(img, vertical_window, mode='same', boundary='symm')
    gradient = np.sqrt(s1 ** 2 + s2 ** 2)
    outline_img = (gradient >= 20) * 255
    plot_task_3(img, s1, s2, gradient, outline_img)
    return outline_img


def plot_task_4(img, laplacian, outline_img):
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(15, 8)
    axs[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title("Исходное изображение")
    axs[0, 1].imshow(laplacian, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title("Оценка лапласиана")
    axs[1, 0].hist(laplacian.flatten(), bins=256, range=[0, 256])
    axs[1, 0].set_title("Гистограмма оценки лапласиана")
    axs[1, 1].imshow(outline_img, cmap='gray', vmin=0, vmax=255)
    axs[1, 1].set_title("Полученные контуры")
    fig.suptitle('Метод согласования для лапласиана')
    plt.show()


def argeed_laplacian(img):
    window = 1 / 3 * np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]])
    laplacian = convolve2d(img, window, mode='same', boundary='symm')
    outline_img = (laplacian >= 25) * 255
    plot_task_4(img, laplacian, outline_img)
    return outline_img


def main():
    # path = '09_lena2.tif'
    path = '01_apc.tif'
    img = read_img(path)
    simple_gradient(img)
    laplacian_approx(img)
    pruitt(img)
    argeed_laplacian(img)


if __name__ == '__main__':
    main()
