import numpy as np


def img_crop(img, size1, size2):
    """
    function to crop an image to size1 along x-axis and size2 along y-axis
    """
    size_x, size_y = img.shape[0], img.shape[1]
    cropped_image = img

    if ((size_x== size1) and (size_y== size2)):

        return img

    elif ((size_x<size1) and (size_y<size2)):
        cropped_image = np.pad(img, (((size1-size_x)//2, (size1-size_x)//2), ((size2-size_y)//2, (size2-size_y)//2)),  "constant")
        return cropped_image

    elif ((size_x<size1)):
        cropped_image = np.pad(img, (((size1-size_x)//2, (size1-size_x)//2), (0, 0)), "constant")


    elif ((size_y<size2)):
        cropped_image = np.pad(img, ((0, 0), ((size2-size_y)//2, (size2-size_y)//2)), "constant")


    size_x, size_y = cropped_image.shape[0], cropped_image.shape[1]

    center_x = size_x // 2
    center_y = size_y // 2

    new_size_x_i = center_x - size1 // 2
    new_size_x_j = center_x + size1 // 2

    new_size_y_i = center_y - size2 // 2
    new_size_y_j = center_y + size2 // 2

    cropped_image = cropped_image[new_size_x_i:new_size_x_j, new_size_y_i:new_size_y_j]

    return cropped_image