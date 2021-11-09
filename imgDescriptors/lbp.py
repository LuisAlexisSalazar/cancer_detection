# ? LBP: Local Binary Pattern
from tools.tool import *
import array
import random


# https://pypi.org/project/netpbmfile/
# https://pypi.org/project/netpbmfile/#description

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
        self.matrix = np.array(self.img.generate_Matrix())
        self.newMatrix = np.copy(self.matrix)

    def algorithm(self, nameFile):
        for idxR in range(self.img.height):
            for idxC in range(self.img.width):
                if idxR == 0 or idxC == 0 or idxC == self.img.width - 1 or idxR == self.img.height - 1:
                    continue
                else:
                    # print(self.newMatrix[idxR, idxC])
                    # print(cal_basic(self.matrix, idxR, idxC))
                    self.newMatrix[idxR, idxC] = bin_to_decimal(cal_basic(self.matrix, idxR, idxC))
        self.generateImg(nameFile)

    def print(self):
        print("Width:", self.img.width)
        print("Height:", self.img.height)
        # 250 arreglos con 202 valores
        # print(len(self.matrix))
        # for sub_matrix in self.matrix:
        #     print(len(sub_matrix))

        print("Valores de la imagen: \n", self.matrix)

    def generateImg(self, nameFile):
        # f = open(self.img.folder + "newImg.pgm", 'wb')
        # f = open("test/" + "newImg.pgm", 'wb')
        # f = open("newImg.pgm", 'wb')
        f = open(nameFile + ".pgm", 'wb')
        pgmHeader = 'P5' + '\n' + str(self.img.width) + ' ' + str(self.img.height) + ' ' + str(255) + '\n'
        pgmHeader_byte = bytearray(pgmHeader, 'utf-8')
        f.write(pgmHeader_byte)
        img = np.reshape(self.newMatrix, (self.img.height, self.img.width))
        for j in range(self.img.height):
            bnd = list(img[j, :])
            f.write(bytearray(bnd))
        f.close()
