# ? BRINT:Binary Rotation Invariant and Noise Tolerant
import matplotlib.pyplot as plt
import numpy as np

from tools.tool import *



def differenceAbsolute(neighbour, center):
    return abs(neighbour - center)


def differences_local(neighbours, center):
    arrayDifferece = []
    for neighbour in neighbours:
        arrayDifferece.append(differenceAbsolute(neighbour, center))

    return arrayDifferece


def localAverage(array, radius, q, ):
    newNeigbours = []
    for i in range(0, 8):
        indexStart = i * radius
        indexFinish = indexStart + q

        localAverageGroup = 0
        for j in range(indexStart, indexFinish):
            localAverageGroup += array[j]

        localAverageGroup = localAverageGroup / q
        # localAverageGroup = localAverageGroup
        newNeigbours.append(localAverageGroup)

    return np.array(newNeigbours)


# ?El valor decimal mínimo de la nueva secuencia binaria se puede obtener girando continuamente la secuencia binaria
def get_min_for_revolve(arr):
    # ?Almacene el valor después de cada turno y finalmente seleccione el que tenga el valor más pequeño
    values = []
    # ?Se utiliza para desplazamiento cíclico, y su correspondiente sistema decimal se calcula respectivamente
    circle = arr * 2
    for i in range(0, 8):
        j = 0
        sum = 0
        bit_sum = 0
        while j < 8:
            sum += circle[i + j] << bit_sum
            bit_sum += 1
            j += 1
        values.append(sum)
    return min(values)


def rule(valueNeigboor, valueCentral):
    if valueNeigboor >= valueCentral:
        return 1
    return 0


def calculateLBP(neighbors, pixelCenter):
    pattern = []
    for neighbor in neighbors:
        pattern.append(rule(neighbor, pixelCenter))
    return pattern


def getUmbral(neighboursReduce):
    umbral = 0
    for neighbour in neighboursReduce:
        umbral += neighbour
    umbral = umbral / 8
    return umbral


def setNeighbours(neighboursReduce, umbral):
    for i in range(0, len(neighboursReduce)):
        neighboursReduce[i] = neighboursReduce[i] - umbral


class Brint:
    def __init__(self, classImg):

        # --ROi de la imagen
        self.matrix = np.array(classImg.matrixRoi)
        self.width = classImg.widthRoi
        self.height = classImg.heightRoi
        self.newMatrixBrintS = np.copy(self.matrix)
        self.newMatrixBrintM = np.copy(self.matrix)

    # def calculate_lbp(self, i, j):
    #     for

    def try_Brint_s(self, radius):
        q = radius
        p = 8 * radius
        # newMatrix = np.zeros([self.height, self.width])
        # revolve_array = np.zeros(self.matrix.shape)

        # !Si es de radio 1 entonces es usar el lbp ya construido
        # !No funcaionaria con radio de 1 pore eso es un caso especial
        for i in range(0, self.height):
            for j in range(0, self.width):
                # Se obtiene el valor mínimo de cada cadena binaria LBP de rotación
                # neighbors = np.array([])
                neighbors = []

                indexI = i
                indexJ = j

                try:
                    # np.append(neighbors, rule(self.matrix[i][j + radius], self.matrix[i][j]))
                    neighbors.append(self.matrix[i][j + radius])
                    # neighbors.append(rule(self.matrix[i][j + radius], self.matrix[i][j]))

                    # -- desde el primer vecino hacia arriba
                    for ascenso_i in range(1, radius + 1):
                        neighbors.append(self.matrix[i - ascenso_i][j + radius])
                        # neighbors.append(rule(self.matrix[i - ascenso_i][j + radius], self.matrix[i][j]))
                        # np.append(neighbors, rule(self.matrix[i - ascenso_i][j + radius], self.matrix[i][j]))
                        # indexI = i - ascenso_i
                        # indexJ = j + radius
                    indexI = i - radius
                    indexJ = j + radius

                    # -- desde la esquina arriba derecha a esquina izquierda
                    displace_left_complete_j = 2 * radius
                    for displace_j in range(1, displace_left_complete_j + 1):
                        neighbors.append(self.matrix[indexI][indexJ - displace_j])
                        # neighbors.append(rule(self.matrix[indexI][indexJ - displace_j], self.matrix[i][j]))
                        # np.append(neighbors, rule(self.matrix[indexI][indexJ - displace_j], self.matrix[i][j]))
                    indexJ = indexJ - displace_left_complete_j
                    # print(len(neighbors))

                    # -- desde esquina izquierda a esquina abajo izquierda
                    displace_left_complete_i = displace_left_complete_j
                    for displace_i in range(1, displace_left_complete_i + 1):
                        neighbors.append(self.matrix[indexI + displace_i][indexJ])
                        # neighbors.append(rule(self.matrix[indexI + displace_i][indexJ], self.matrix[i][j]))
                        # np.append(neighbors, rule(self.matrix[indexI + displace_i][indexJ], self.matrix[i][j]))
                    indexI = indexI + displace_left_complete_i
                    # print(len(neighbors))
                    # -- desde esquina abajo izquierda a esquina derecha abajo
                    displace_left_complete_j = 2 * radius
                    for displace_j in range(1, displace_left_complete_j + 1):
                        neighbors.append(self.matrix[indexI][indexJ + displace_j])
                        # neighbors.append(rule(self.matrix[indexI][indexJ + displace_j], self.matrix[i][j]))
                        # np.append(neighbors, rule(self.matrix[indexI][indexJ + displace_j], self.matrix[i][j]))
                    indexJ = indexJ + displace_left_complete_j

                    # -- desde esquina abajo derecha a antes de la mitad de punto medio
                    # !Revisar
                    # displace_left_complete_i = radius
                    displace_left_complete_i = displace_left_complete_i // 2
                    for displace_i in range(1, displace_left_complete_i):
                        neighbors.append(self.matrix[indexI - displace_i][indexJ])
                        # neighbors.append(rule(self.matrix[indexI - displace_i][indexJ], self.matrix[i][j]))
                        # np.append(neighbors, rule(self.matrix[indexI - displace_i][indexJ], self.matrix[i][j]))

                    # ?Reducción de dimensionalidad
                    neighbors = localAverage(neighbors, radius, q)
                    print(neighbors)
                    # pattern = calculateLBP(neighbors, self.matrix[i][j])
                    # print(pattern)

                    # ? la nueva matrix
                    # self.newMatrix[i, j] = bin_to_decimal(pattern)

                except IndexError:
                    continue
            # revolve_key = get_min_for_revolve(sum)

    def Brint_s(self, radius, q_points):
        points = 8
        n_points = q_points * points
        # height = src.shape[0]
        # width = src.shape[1]
        # dst= src.copy()

        self.matrix.astype(dtype=np.float32)
        self.newMatrixBrintS.astype(dtype=np.float32)

        # neighbours = np.zeros((1, n_points), dtype=np.uint8)
        neighbours = np.zeros(n_points, dtype=np.uint8)
        # lbp_value = np.zeros((1, n_points), dtype=np.uint8)
        lbp_value = np.zeros(n_points, dtype=np.uint8)
        for x in range(radius, self.width - radius - 1):
            for y in range(radius, self.height - radius - 1):
                lbp = 0.
                # Primero se calcula el valor de píxel correspondiente a un total de n_points
                # puntos, usando interpolación bilineal
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

                # --Reducción de dimensionalidad
                neighbours = localAverage(neighbours, radius, q_points)
                neighbours = neighbours.astype(int)
                # -----------------------------

                # for n in range(n_points):
                # for n in range(n_points):
                for n in range(points):
                    if neighbours[n] > center:
                        lbp_value[n] = 1
                    else:
                        lbp_value[n] = 0

                for n in range(points):
                    lbp += lbp_value[n] * 2 ** n

                neighbours = np.zeros(n_points, dtype=np.uint8)

                # !falta min ROR
                self.newMatrixBrintS[y, x] = int(lbp / (2 ** points - 1) * 255)

    def Brint_m(self, radius, q_points):
        points = 8
        n_points = q_points * points
        # height = src.shape[0]
        # width = src.shape[1]
        # dst= src.copy()

        self.matrix.astype(dtype=np.float32)
        self.newMatrixBrintM.astype(dtype=np.float32)

        # neighbours = np.zeros((1, n_points), dtype=np.uint8)
        neighbours = np.zeros(n_points, dtype=np.uint8)
        # lbp_value = np.zeros((1, n_points), dtype=np.uint8)
        lbp_value = np.zeros(n_points, dtype=np.uint8)
        for x in range(radius, self.width - radius - 1):
            for y in range(radius, self.height - radius - 1):
                lbp = 0.
                # Primero se calcula el valor de píxel correspondiente a un total de n_points
                # puntos, usando interpolación bilineal
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

                # print("neighbours -> ", neighbours)
                # print("Central -> ", center)
                neighboursDifferecneLocales = differences_local(neighbours, center)
                # print("neighboursDifferecneLocales -> ", neighboursDifferecneLocales)

                # --Reducción de dimensionalidad
                # zrqn
                neighbours = localAverage(neighboursDifferecneLocales, radius, q_points)
                neighbours = neighbours.astype(int)
                # print("neighbours Redcue dimensionalidad -> ", neighbours)
                # -----------------------------
                umbral = getUmbral(neighbours)
                # print("umbral -> ", umbral)
                setNeighbours(neighbours, umbral)
                # print("Nuevos vecinos -> ", neighbours)
                for n in range(points):
                    if neighbours[n] > center:
                        lbp_value[n] = 1
                    else:
                        lbp_value[n] = 0
                # print("Central -> ", center)
                # print("LBP -> ", lbp_value, end="\n\n")
                for n in range(points):
                    lbp += lbp_value[n] * 2 ** n

                neighbours = np.zeros(n_points, dtype=np.uint8)

                self.newMatrixBrintM[y, x] = int(lbp / (2 ** points - 1) * 255)

    def genereteImg(self, descriptor):
        plt.subplot(121), plt.imshow(self.matrix, cmap='gray')
        plt.title('Imagen con ROI'), plt.xticks([]), plt.yticks([])

        if descriptor == "BrintS":
            plt.subplot(122), plt.imshow(self.newMatrixBrintS, cmap='gray')
            plt.title(descriptor), plt.xticks([]), plt.yticks([])
        elif descriptor == "BrintM":
            plt.subplot(122), plt.imshow(self.newMatrixBrintM, cmap='gray')
            plt.title(descriptor), plt.xticks([]), plt.yticks([])
        plt.show()

    def showHistograms(self):
        names = ['Brint S', 'Brint M']

        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(1, 1, 1)
        # n, bins, patches = ax1.hist(self.newMatrixBrintS)
        #
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(1, 1, 1)
        # n, bins, patches = ax2.hist(self.newMatrixBrintM)
        #
        # plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist(self.newMatrixBrintS, bins='auto')
        ax2.hist(self.newMatrixBrintM, bins='auto')
        ax1.set_title(names[0])
        ax2.set_title(names[1])
        plt.show()
