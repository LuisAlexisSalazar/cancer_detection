# ? LBP: Local Binary Pattern
from tools.tool import *

# https://www.fatalerrors.org/a/implementation-of-lbp-algorithm-in-python.html


def rule(valueNeigboor, valueCentral):
    if valueNeigboor >= valueCentral:
        return 1
    return 0


def cal_basic(img, i, j):
    sum = []
    pixelCenter = img[i][j]
    sum.append(rule(img[i - 1, j], pixelCenter))
    sum.append(rule(img[i - 1, j + 1], pixelCenter))
    sum.append(rule(img[i, j + 1], pixelCenter))
    sum.append(rule(img[i + 1, j + 1], pixelCenter))
    sum.append(rule(img[i + 1, j], pixelCenter))
    sum.append(rule(img[i + 1, j - 1], pixelCenter))
    sum.append(rule(img[i, j - 1], pixelCenter))
    sum.append(rule(img[i - 1, j - 1], pixelCenter))

    return sum


# ? window 3x3
class Lbp:

    def __init__(self, classImg):
        self.img = classImg

        # --Imagen Original
        # self.matrix = np.array(self.img.matrixOriginal)
        # self.width = classImg.width
        # self.height = classImg.height
        # --ROi de la imagen
        self.matrix = np.array(self.img.matrixRoi)
        # print("Shape de la matrix:", self.matrix.shape)
        # 1024-650 -> Indices (1023-648)
        self.width = classImg.widthRoi
        self.height = classImg.heightRoi

        # !Borrar
        # print(self.matrix[0][650])
        # for i in self.matrix:
        #     print(len(i))

        # --Matrix resultante de LBP
        # self.newMatrix = np.copy(self.matrix)
        self.newMatrix = np.zeros([self.height, self.width])

    def algorithm(self):
        for idxR in range(self.height):  # 1024
            for idxC in range(self.width):  # 650
                if idxR == 0 or idxC == 0 or idxR == self.height - 1 or idxC == self.width - 1:
                    continue
                else:

                    # print(self.newMatrix[idxR, idxC])
                    # print(cal_basic(self.matrix, idxR, idxC))
                    self.newMatrix[idxR, idxC] = bin_to_decimal(cal_basic(self.matrix, idxR, idxC))

    def generateHistogram(self):
        # !Necesario para "calcHist" de Float64 a Float32
        self.newMatrix = np.float32(self.newMatrix)

        hist = cv2.calcHist(self.newMatrix, [0], None, [256], [0, 256])
        plt.plot(hist, color='r')
        plt.title('Histograma en escala a grises')
        plt.show()

    def generateImg(self, nameFile):
        # ! No es necesario pero interesante para otras funciones
        # self.newMatrix = np.float32(self.newMatrix)

        # f = open(nameFile + ".pgm", 'wb')
        # pgmHeader = 'P5' + '\n' + str(self.img.width) + ' ' + str(self.img.height) + ' ' + str(255) + '\n'
        # pgmHeader_byte = bytearray(pgmHeader, 'utf-8')
        # f.write(pgmHeader_byte)
        # img = np.reshape(self.newMatrix, (self.img.height, self.img.width))

        # --------------------
        plt.subplot(121), plt.imshow(self.img.matrixOriginal, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(self.newMatrix, cmap='gray')
        plt.title('LBP and ROI'), plt.xticks([]), plt.yticks([])

        plt.show()
        # -------------------------------

        # for j in range(self.img.height):
        #     bnd = list(img[j, :])
        #     f.write(bytearray(bnd))
        # f.close()
