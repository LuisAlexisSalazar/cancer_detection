# coding: utf8
import time
from datetime import timedelta
import math
import numpy as np


class Img:
    matrixRoi = None
    widthRoi = None
    heightRoi = None

    def __init__(self, name_File, folder="dataSet/"):
        try:
            self.pgmFile = open(folder + name_File, 'rb')
            self.name_File = name_File
            self.folder = folder
        except IOError:
            print("Archivo no encontrado")
            exit()

        header = self.pgmFile.readline()
        assert header[:2] == b'P5'

        if self.isGimp():
            self.pgmFile.readline()

        (self.width, self.height) = [int(i) for i in self.pgmFile.readline().split()]

        self.depth = int(self.pgmFile.readline())
        assert self.depth <= 255

        self.matrixOriginal = self.generate_Matrix()

    def generate_Matrix(self):
        matrix = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(ord(self.pgmFile.read(1)))
            matrix.append(row)
        return matrix

    def isGimp(self):
        temp_pgmf = open(self.folder + self.name_File, 'rb')
        temp_pgmf.readline()
        temp_line = temp_pgmf.readline()
        if temp_line == b'# Created by GIMP version 2.10.12 PNM plug-in\n':
            return True
        return False

    def print_dimension_both_img(self):
        print("Widht R:", self.widthRoi)
        print("Height R:", self.heightRoi)

        print("Widht O:", self.width)
        print("Height O:", self.height)


def getLabels(amount=330):
    # pathLabels = open("../dataSet/Info.txt")
    labels = []
    pathLabels = open("dataSet/Info.txt")
    with pathLabels as f:
        lines = f.readlines()
    # 104-433 es lo unico que nos importa

    lines = lines[103:433]
    count_normal = 0
    count_maligno = 0
    amountTotalRead = 0
    #  Nomrla -> 0 y Maligno -> 1
    for line in lines:
        # print(type(line))
        list_split = line.split(sep=" ")

        if list_split[2] == 'NORM':
            # labels.append("N")
            labels.append(0)
            count_normal += 1
        else:
            # labels.append("M")
            labels.append(1)
            count_maligno += 1

        amountTotalRead += 1
        if amountTotalRead == amount:
            break

    print("Normales -> ", count_normal)
    print("Maligno -> ", count_maligno)
    return np.array(labels)


def bin_to_decimal(bin):
    res = 0
    bit_num = 0
    for i in bin[::-1]:
        res += i << bit_num
        bit_num += 1
    return res


# days, hours, minutes,
def start_time_measure(message=None):
    if message:
        print(message)
    return time.time()


def end_time_measure(start_time):
    end_time = time.time()
    return end_time - start_time


# function s or g
def getPattern(value):
    if value >= 0:
        return 1
    else:
        return 0


def _bit_rotate_right(value, length):
    """Cyclic bit shift to the right.
    Parameters
    ----------
    value : int
        integer value to shift
    length : int
        number of bits of integer
    """
    return (value >> 1) | ((value & 1) << (length - 1))
