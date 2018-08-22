import numpy as np 

def convolver(image, kernel):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return new_image
im = np.array([[1, 2, 0, 0], [5, 3, 0, 4], [0, 0, 0, 7], [9, 3, 0, 0]])
kernel = np.array([[1,1,1],[1,1,0],[1,0,0]])

result = convolver(im, kernel)

print(result)