import numpy as np
from skimage.util import random_noise
from scipy.signal import convolve2d
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt


def get_plot(source_img, gause, noisy_img, linear_filter, median_filter, suptitle):
    fig, axs = plt.subplots(3, 3)
    fig.set_size_inches(w=17, h=9.5)
    fig.suptitle(suptitle)
    axs[0, 0].imshow(source_img, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title("Исходное изображение")

    axs[0, 1].imshow(gause, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title("Шум")

    axs[0, 2].imshow(noisy_img, cmap='gray', vmin=0, vmax=255)
    axs[0, 2].set_title("Зашумленное изображение")

    axs[1, 0].imshow(source_img, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].set_title("Исходное изображение")

    axs[1, 1].imshow(noisy_img, cmap='gray', vmin=0, vmax=255)
    axs[1, 1].set_title("Зашумленное изображение")

    axs[1, 2].imshow(linear_filter, cmap='gray', vmin=0, vmax=255)
    axs[1, 2].set_title("Восстановленное изображение после линейной фильтрации")

    axs[2, 0].imshow(source_img, cmap='gray', vmin=0, vmax=255)
    axs[2, 0].set_title("Исходное изображение")

    axs[2, 1].imshow(noisy_img, cmap='gray', vmin=0, vmax=255)
    axs[2, 1].set_title("Зашумленное изображение")

    axs[2, 2].imshow(median_filter, cmap='gray', vmin=0, vmax=255)
    axs[2, 2].set_title("Восстановленное изображение после медианной фильтрации")

    plt.show()


def generate_chess_board(n):
    img = np.zeros((n, n), dtype=int)
    for i in range(0, n, int(n / 8)):
        if int((i / (n / 8)) % 2) != 0:
            for j in range(0, n, int(n / 8)):
                if (j / (n / 8) % 2) != 0:
                    img[i: int(i + n / 8), j:int(j + n / 8)] = 160
                else:
                    img[i: int(i + n / 8), j:int(j + n / 8)] = 96
        else:
            for j in range(0, n, int(n / 8)):
                if (j / (n / 8) % 2) != 0:
                    img[i: int(i + n / 8), j:int(j + n / 8)] = 96
                else:
                    img[i: int(i + n / 8), j:int(j + n / 8)] = 160
    return img


def calc_SKO(img, recovered_img):
    res = 0
    m, n = img.shape[0], img.shape[1]
    for i in range(0, m):
        for j in range(0, n):
            res += (recovered_img[i, j] - img[i, j]) ** 2
    res /= m * n
    return res


def linear_filter(noisy_img):
    window = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    recovery_img = convolve2d(noisy_img, window, mode='same')
    return recovery_img


def task_1(img, n):
    d_x = np.var(img)
    d_v = d_x / n
    noise = np.random.normal(0, np.sqrt(d_v), (img.shape[0], img.shape[1]))
    noisy_img = img + noise
    eps_noise = calc_SKO(img, noisy_img)
    recovered_img_lin = linear_filter(noisy_img)
    eps_lin_filter = calc_SKO(img, recovered_img_lin)
    k_linear = eps_lin_filter / eps_noise
    medians_filter = median_filter(noisy_img, size=5)
    eps_medians_filter = calc_SKO(medians_filter, img)
    k_median = eps_medians_filter / eps_noise
    get_plot(img, noise + 128, noisy_img, recovered_img_lin, medians_filter, f"Белый шум (сигнал/шум = {n})")
    print(f"Дисперсия изображения:{d_x}")
    print(f'Дисперсия адитивного белого шума("сигнал/шум" = {n}) = {d_v}')
    print(f"СКО линейной фильтрации: {eps_lin_filter}")
    print(f"Коэффициент линейной фильтрации: {k_linear}")
    print(f"СКО медианной фильтрации: {eps_medians_filter}")
    print(f"Коэффициент медианной фильтрации: {k_median}")
    print('-----------------------------------------------------------------------------------------------------------')


def task_2(img, p):
    d_x = np.var(img)
    norm_img = img / 255
    noisy_img = random_noise(norm_img, mode='s&p', amount=p)
    noisy_img *= 255
    impulse = np.copy(noisy_img)
    impulse[impulse == 96] = 128
    impulse[impulse == 160] = 128
    d_v = np.var(impulse)
    eps_noise = calc_SKO(img, noisy_img)
    recovered_img_lin = linear_filter(noisy_img)
    eps_lin_filter = calc_SKO(img, recovered_img_lin)
    k_linear = eps_lin_filter / eps_noise
    medians_filter = median_filter(noisy_img, size=5)
    eps_medians_filter = calc_SKO(medians_filter, img)
    k_median = eps_medians_filter / eps_noise
    get_plot(img, impulse, noisy_img, recovered_img_lin, medians_filter, f"Ипульсный шум (вероятность = {p})")
    print(f"Дисперсия изображения:{d_x}")
    print(f'Дисперсия ипульснго шум шума(вероятность = {p}) = {d_v}')
    print(f"СКО линейной фильтрации: {eps_lin_filter}")
    print(f"Коэффициент линейной фильтрации: {k_linear}")
    print(f"СКО медианной фильтрации: {eps_medians_filter}")
    print(f"Коэффициент медианной фильтрации: {k_median}")
    print('-----------------------------------------------------------------------------------------------------------')


def main():
    img = generate_chess_board(128)
    task_1(img, 1)
    task_1(img, 10)
    task_2(img, 0.1)
    task_2(img, 0.3)
    print("Водянов")


if __name__ == '__main__':
    main()
