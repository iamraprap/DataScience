import numpy as np 

def max_pool(image, k):
    new_image = np.zeros((image.shape[1],image.shape[0]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            max = 0
            for x in range(i,i+2 if i+2<image.shape[0] else image.shape[0]):
                for y in range(j,j+2 if j+2<image.shape[1] else image.shape[1]):
                    max = max if max>=image[x][y] else image[x][y]
                    print("x:%d, y:%d, img:%d" % (x, y, image[x][y]))
            print()
            new_image[i][j] = max

    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            print("i:%d, j:%d, nimage:%d" % (i, j, new_image[i][j]))
    return np.array(new_image)
    
im = np.array([[1, 2, 0, 0], [5, 3, 0, 4], [0, 0, 0, 7], [9, 3, 0, 0]])
k = 2

#result = max_pool(im, kernel)
result = max_pool(im, k)
print(result)