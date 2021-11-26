from imgDescriptors.lbp import Lbp
from tools.tool import *
import glob
from segmeter.roi import *
from imgDescriptors.brint import Brint
from imgDescriptors.rilqp import Rilqp
import pandas as pd
from classification.svm import Svm
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern


# ctrl + shift + i show definición
# ctrl + B : travel to implementation


# RILQP and BRINT
def descriptor(nameDescriptor, amount_Total):
    folder = "dataSet/"
    labels = getLabels(amount_Total)
    processTime = []
    path = glob.glob(folder + "*.pgm")
    nameFile = []

    # --entrada para el clasificador
    img_arrays = []

    for index, path_img in enumerate(path):
        nameFile = path_img.split("\\")[1]
        img = Img(nameFile, folder)
        get_matrix_ROI(img)
        imgs_lbp = []

        total_start_time = start_time_measure()

        # --Descriptores nuevos de LBP
        # img_lbp = local_binary_pattern(img.matrixRoi, 8, 1)
        # img_lbp = local_binary_pattern(img.matrixRoi, 8, 1)
        if nameDescriptor == "BRINT":
            BRINT = Brint(img)
            # BRINT.Brint_m(radius=2, q_points=2)
            BRINT.Brint_s(radius=2, q_points=2)
            # imgs_lbp = [BRINT.newMatrixBrintM, BRINT.newMatrixBrintS]
            img_lbp = BRINT.newMatrixBrintS
        elif nameDescriptor == "RILQP":
            RILQ = Rilqp(img)
            RILQ.algorithm(radius=2, n_points=8, tau1=2, tau2=5)
            # imgs_lbp = [RILQ.newMatrixA, RILQ.newMatrixB, RILQ.newMatrixC, RILQ.newMatrixD]
            img_lbp = RILQ.newMatrixA
        else:
            img_lbp = local_binary_pattern(img.matrixRoi, 8, 1)

        processTime.append(end_time_measure(total_start_time))

        # img_lbp = []
        # for img_descriptor in imgs_lbp:
        #     img_lbp = img_descriptor.flatten()

        img_arrays.append(img_lbp.flatten())

        # img_arrays.append(img_lbp.flatten())
        print("Imagen -> " + nameFile + " terminada")
        if index == amountTotal - 1:
            break

    print("Tiempos del descriptor :", processTime)
    print("Tiempo Promedio del descriptor " + nameDescriptor + ":", sum(processTime) / len(processTime))

    df = pd.DataFrame(img_arrays)
    df['Target'] = labels
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)
    print('Splitted Successfully')

    total_start_time = start_time_measure()
    svm = Svm(x_train, y_train)
    timeTrain = end_time_measure(total_start_time)
    print("Tiempo de entrenamiento con el descriptor" + nameDescriptor + ":", timeTrain)
    svm.analyzeMetric(x_test, y_test)


if __name__ == '__main__':

    amountTotal = 50
    descriptor("BRINT", amountTotal)
    descriptor("RILQP", amountTotal)

