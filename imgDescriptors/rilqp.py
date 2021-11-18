# ? RILQP: Rotation Invariant Local Quinary Patterns
import matplotlib.pyplot as plt
import numpy as np
from tools.tool import *


# función d(x)
def ruleLQP(n, t1, t2):
    if t2 <= n:
        return 2
    elif t1 <= n < t2:
        return 1
    elif -t1 <= n < t1:
        return 0
    elif -t2 <= n < -t1:
        return -1
    elif n < -t2:
        return -2


def pattternLQP(neighbours, center, t1, t2):
    pattern = []
    for neighbour in neighbours:
        pattern.append(ruleLQP(neighbour - center, t1, t2))
    return pattern


def g_i(p_i, value):
    if p_i == value:
        return 1
    else:
        return 0


def splitPattern(pattern):
    fourPattern = []
    tempPattern = []
    # --g1
    for p_i in pattern:
        tempPattern.append(g_i(p_i, 2))
    fourPattern.append(tempPattern.copy())
    tempPattern.clear()

    # --g2
    for p_i in pattern:
        tempPattern.append(g_i(p_i, 1))
    fourPattern.append(tempPattern.copy())
    tempPattern.clear()

    # --g3
    for p_i in pattern:
        tempPattern.append(g_i(p_i, -1))
    fourPattern.append(tempPattern.copy())
    tempPattern.clear()

    # --g4
    for p_i in pattern:
        tempPattern.append(g_i(p_i, -2))
    fourPattern.append(tempPattern.copy())
    tempPattern.clear()

    return fourPattern


class Rilqp:
    def __init__(self, classImg):
        # --ROi de la imagen
        self.matrix = np.array(classImg.matrixRoi)
        self.width = classImg.widthRoi
        self.height = classImg.heightRoi
        self.newMatrixA = np.copy(self.matrix)
        self.newMatrixB = np.copy(self.matrix)
        self.newMatrixC = np.copy(self.matrix)
        self.newMatrixD = np.copy(self.matrix)

    # Paper Ejemplo de parametrización
    # ?P=10 r=1 t1=2 t2=5
    # ?P=18 r=2 t1=2 t2=5
    def algorithm(self, radius, n_points, tau1=2, tau2=5):
        self.matrix.astype(dtype=np.float32)
        self.newMatrixA.astype(dtype=np.float32)
        self.newMatrixB.astype(dtype=np.float32)
        self.newMatrixC.astype(dtype=np.float32)
        self.newMatrixD.astype(dtype=np.float32)

        neighbours = np.zeros(n_points, dtype=np.uint8)
        lbp_value = np.zeros(n_points, dtype=np.uint8)

        for x in range(radius, self.width - radius - 1):
            for y in range(radius, self.height - radius - 1):
                lbp = 0.
                for n in range(n_points):
                    theta = float(2 * np.pi * n) / n_points
                    x_n = x + radius * np.cos(theta)
                    y_n = y - radius * np.sin(theta)

                    # 向下取整
                    x1 = int(math.floor(x_n))
                    y1 = int(math.floor(y_n))
                    # 向上取整
                    x2 = int(math.ceil(x_n))
                    y2 = int(math.ceil(y_n))

                    # 将坐标映射到0-1之间
                    tx = np.abs(x - x1)
                    ty = np.abs(y - y1)

                    # 根据0-1之间的x，y的权重计算公式计算权重
                    w1 = (1 - tx) * (1 - ty)
                    w2 = tx * (1 - ty)
                    w3 = (1 - tx) * ty
                    w4 = tx * ty

                    # 根据双线性插值公式计算第k个采样点的灰度值
                    neighbour = self.matrix[y1, x1] * w1 + self.matrix[y2, x1] * w2 + self.matrix[y1, x2] * w3 + \
                                self.matrix[y2, x2] * w4

                    neighbours[n] = neighbour

                center = self.matrix[y, x]

                # --Local Ternary Pattern LQP
                pattern = pattternLQP(neighbours, center, tau1, tau2)
                FourPattern = splitPattern(pattern)
                # ----

                for i, pattern_n in enumerate(FourPattern):
                    for n in range(n_points):
                        lbp += pattern_n[n] * 2 ** n
                    if i == 0:
                        self.newMatrixA[y, x] = int(lbp / (2 ** n_points - 1) * 255)
                    elif i == 1:
                        self.newMatrixB[y, x] = int(lbp / (2 ** n_points - 1) * 255)
                    elif i == 2:
                        self.newMatrixC[y, x] = int(lbp / (2 ** n_points - 1) * 255)
                    elif i == 3:
                        self.newMatrixD[y, x] = int(lbp / (2 ** n_points - 1) * 255)

                neighbours = np.zeros(n_points, dtype=np.uint8)

    def genereteImgParte1(self):
        plt.subplot(121), plt.imshow(self.newMatrixA, cmap='gray')
        plt.title('G1 (x) == 2'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(self.newMatrixB, cmap='gray')
        plt.title('G2 (x) == 1'), plt.xticks([]), plt.yticks([])
        plt.show()

    def genereteImgParte2(self):
        plt.subplot(122), plt.imshow(self.newMatrixC, cmap='gray')
        plt.title('G3 (x) == -1'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(self.newMatrixD, cmap='gray')
        plt.title('G4 (x) == -2'), plt.xticks([]), plt.yticks([])
        plt.show()

    def genereteImg(self):
        f, axarr = plt.subplots(2, 2, figsize=(10, 10))

        axarr[0, 0].set_title('G1 -> 2', fontsize=15)
        axarr[0, 0].imshow(self.newMatrixA, cmap='gray')

        axarr[0, 1].set_title('G2 -> 1', fontsize=15)
        axarr[0, 1].imshow(self.newMatrixB, cmap='gray')

        axarr[1, 0].set_title('G3 -> -1', fontsize=15)
        axarr[1, 0].imshow(self.newMatrixC, cmap='gray')

        axarr[1, 1].set_title('G4 -> -2', fontsize=15)
        axarr[1, 1].imshow(self.newMatrixD, cmap='gray')
        plt.show()
