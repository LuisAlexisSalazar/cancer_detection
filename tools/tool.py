import numpy as np


# pgm-reader 0.0.1


class Img:
    def __init__(self, name_File, folder="dataSet/"):
        try:
            self.pgmf = open(folder + name_File, 'rb')
            self.name_File = name_File
            self.folder = folder
        except IOError:
            print("Archivo no encontrado")
            exit()

        header = self.pgmf.readline()
        assert header[:2] == b'P5'

        if self.isGimp():
            self.pgmf.readline()

        (self.width, self.height) = [int(i) for i in self.pgmf.readline().split()]

        self.depth = int(self.pgmf.readline())
        assert self.depth <= 255

    def generate_Matrix(self):
        matrix = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(ord(self.pgmf.read(1)))
            matrix.append(row)

        # for i in matrix:
        #     print(len(i))
        return matrix

    def isGimp(self):
        temp_pgmf = open(self.folder + self.name_File, 'rb')
        temp_pgmf.readline()
        temp_line = temp_pgmf.readline()
        if temp_line == b'# Created by GIMP version 2.10.12 PNM plug-in\n':
            return True
        return False


def bin_to_decimal(bin):  # Binary to decimal
    res = 0
    bit_num = 0  # Shift left
    for i in bin[::-1]:
        res += i << bit_num  # Shifting n bits to the left is equal to multiplying by 2 to the nth power
        bit_num += 1
    return res
