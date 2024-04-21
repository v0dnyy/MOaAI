import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
from lab_1 import thresholding


def get_plot(noise, img, R_contraste, R_res):
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(w=17, h=7)

    axs[0].imshow(noise, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Шум")

    axs[1].imshow(img, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title("Изображение с бъектами")

    axs[2].imshow(R_contraste, cmap='gray', vmin=0, vmax=255)
    axs[2].set_title("Корреляционное поле")

    axs[3].imshow(R_res, cmap='gray', vmin=0, vmax=255)
    axs[3].set_title("Результат пороговой обработки")

    plt.show()


def generate_noise(n, d_v=100):
    img = np.full((n, n), 128, dtype=int)
    noise = np.random.normal(0, np.sqrt(d_v), (img.shape[0], img.shape[1])).astype(np.int32)
    img += noise
    return img


def add_obg(image, object_mask, count):
    img = image.copy()
    positions = np.random.randint(low=[0, 0],
                                  high=[img.shape[0] - object_mask.shape[0], img.shape[1] - object_mask.shape[1]],
                                  size=(count, 2))
    obj_mask = object_mask.copy() * 255
    for position in positions:
        img[position[0]:(position[0] + obj_mask.shape[0]), position[1]:(position[1] + obj_mask.shape[1])] += obj_mask

    return img


def correlator(noise, img, obg_mask):
    R = convolve2d(img / (np.std(img) * len(img)), np.flip(obg_mask) / (np.std(np.flip(obg_mask)) * len(obg_mask)),
                   mode='same', boundary='symm')
    f_min = np.min(R)
    f_max = np.max(R)
    g_min = 0
    g_max = 255
    a = (g_max - g_min) / (f_max - f_min)
    b = (g_min * f_max - g_max * f_min) / (f_max - f_min)
    R_contrast = a * R + b
    res = thresholding(R, 0.82)
    get_plot(noise, img, R_contrast, res)


def main():
    object_1 = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]])
    object_2 = np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]])
    noise = generate_noise(64, 100)
    image_with_same_object = add_obg(noise, object_1, 3)
    image_with_different_object = add_obg(image_with_same_object, object_2, 3)
    correlator(noise, image_with_same_object, object_1)
    correlator(noise, image_with_different_object, object_1)
    print()


if __name__ == '__main__':
    main()
