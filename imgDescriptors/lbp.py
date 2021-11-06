# ? LBP: Local Binary Pattern
from cv2 import cv2
import numpy as np


def isGimp(name_File):
    temp_pgmf = open("test/" + name_File, 'rb')
    temp_pgmf.readline()
    temp_line = temp_pgmf.readline()
    if temp_line == b'# Created by GIMP version 2.10.12 PNM plug-in\n':
        return True
    return False


class Lbp:
    def __init__(self, name_File):
        # self.matrix_img = open("test/" + name_File, 'rb')
        # self.pgmf = open("dataSet/" + name_File, 'rb')
        try:
            self.pgmf = open("test/" + name_File, 'rb')
            self.name_File = name_File
        except IOError:
            print("Archivo no encontrado")
            return

        header = self.pgmf.readline()
        assert header[:2] == b'P5'

        if isGimp(self.name_File):
            self.pgmf.readline()

        (self.width, self.height) = [int(i) for i in self.pgmf.readline().split()]

        self.depth = int(self.pgmf.readline())
        assert self.depth <= 255

        self.matrix = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(ord(self.pgmf.read(1)))
            self.matrix.append(row)

    def print(self):
        print("Width:", self.width)
        print("Height:", self.height)
        # print(self.raster)
        my_array = np.array(self.matrix)
        print("Valores de la imagen: \n",my_array)

        # cv2.imshow("Image", self.matrix_img)
        # cv2.waitKey(0)
