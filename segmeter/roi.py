# https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
# https://www.youtube.com/watch?v=3T3XpXxNAV4
# https://programmerclick.com/article/16741294597/
# Dibujar el rectangulo con imagenes grises es decir trasnformarlo en 3 canales
# https://stackoverflow.com/questions/61243800/how-to-draw-a-red-shape-on-a-black-background-in-cv2
# Recordar que el 0,0 es esquina de arriba izquierda
#  Punto arriba derecha (769,0)
#  Punto abajo  derecha (769,1024)
#
#  Punto arriba izquierda (187,0)
#  Punto abajo  izquierda (187,1024)

# Rectangulo -> 2 puntos top left y bot right
# Segmento ROI simplemnente el ancho de la imagen y alto desde
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

ptl = (187, 0)
# pbr = (769, 1024)
pbr = (837, 1024)


# !nota: esta mal el valor de x de la parte de la izquierda

def get_matrix_ROI(Img):
    # img = img
    # img = np.uint8(Img.matrixOriginal)
    img = Img.matrixOriginal

    # colores variados
    # img = cv2.merge([img, img, img])

    # ?ROI: región de interes -> primero ajustar bien los puntos
    imgRoi = np.copy(img)
    # imgRoi = imgRoi[ptl[0]:pbr[0], ptl[1]:pbr[1]]
    imgRoi = imgRoi[ptl[1]:pbr[1], ptl[0]:pbr[0]]

    # --color white
    # cv2.rectangle(img, ptl, pbr, (2**16, 0, 0), 5)
    # -- colores variados
    # cv2.rectangle(img, ptl, pbr, (255, 0, 0), 2)

    # --Dibujar la imagen original y el ROI
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(imgRoi, cmap='gray')
    # plt.title('ROI'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # --Dibujar una sola imagen
    # cv2.imshow('Img', imgRoi)
    # cv2.waitKey(0)


    # --Configurar los atributos de la clase Imagen
    Img.matrixRoi = imgRoi
    Img.heightRoi = imgRoi.shape[0]
    Img.widthRoi = imgRoi.shape[1]

