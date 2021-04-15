from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.ensemble import IsolationForest
import cv2
import os
from sklearn.ensemble import IsolationForest

#configuracion necesaria de pyplot para ver las imagenes en escala de grises
plt.rcParams['image.cmap'] = 'gray'

def pixRelevant(raw, mask):
    """
    Función que devuelve las áreas relevantes para el análisis mediante aplicar una máscara
    """
    #Convertimos todos los pixelex de la máscara en 0 o 1
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
              mask[i][j] = 1
    relevant_img = raw * mask
    return relevant_img

def numLabels(img_matrix):
    """
    Función que devuelve el número de etiquetas que usó el experto para
    catalogar las diferentes zonas en el pulmón, se cuenta la zona
    negra como una etiqueta.
    """
    labels_exp = []
    for i in range(img_matrix.shape[0]):
        for j in range(img_matrix.shape[1]):
            if img_matrix[i][j]  not in labels_exp:
                labels_exp.append(img_matrix[i][j])
    return len(labels_exp)

def createAttrImg(img,
                  kernel_mean = 10,
                  kernel_gauss = 0,
                  kernel_median = 9,
                  canny_min = 100,
                  canny_max = 500,
                  d_bilateral = 10,
                  sigmaColor_bilateral = 5,
                  sigmaSpace_bilateral = 5):

    """
    Función que devuelve un objeto de tipo DataFrame, este contiene como filas a cada uno de los pixeles de la imagen.
    cada atributo representa un filtro aplicado en la imagen.
    """
    #Sin filtro
    img_inline = np.reshape(img, (1, img.shape[0]*img.shape[1])).squeeze()
    df_img_res = pd.DataFrame(img_inline, columns=['Raw'])


    #Filtro Medio
    image_test = cv2.blur(img, (kernel_mean, kernel_mean))
    df_img_res['Mean'] = np.reshape(image_test, (1, img.shape[0]*img.shape[1])).squeeze()

    #Filtro Gaussiano
    image_test = cv2.GaussianBlur(img, (kernel_gauss, kernel_gauss), cv2.BORDER_DEFAULT)
    df_img_res['Gauss'] = np.reshape(image_test, (1, img.shape[0]*img.shape[1])).squeeze()

    #Filtro Mediano
    image_test = cv2.medianBlur(img, kernel_median)
    df_img_res['Median'] = np.reshape(image_test, (1, img.shape[0]*img.shape[1])).squeeze()

    #Filtro Conny
    image_test = cv2.Canny(img, canny_min, canny_max)
    df_img_res['Conny'] = np.reshape(image_test, (1, img.shape[0]*img.shape[1])).squeeze()

    #Filtro Horizontal
    kernel = np.array([[-1.0, -1.0, -1.0],
                   [2.0, 2.0, 2.0],
                   [-1.0, -1.0, -1.0]])
    image_test = cv2.filter2D(img, -1, kernel)
    df_img_res['Horizontal'] = np.reshape(image_test, (1, img.shape[0]*img.shape[1])).squeeze()

    #Filtro Vertical
    kernel = np.array([[-1.0, 2.0, -1.0],
                   [-1.0, 2.0, -1.0],
                   [-1.0, 2.0, -1.0]])
    image_test = cv2.filter2D(img, -1, kernel)
    df_img_res['Vertical'] = np.reshape(image_test, (1, img.shape[0]*img.shape[1])).squeeze()

    #Filtro Laplaciano
    kernel = np.array([[0.0, -1.0, 0.0],
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])
    image_test = cv2.filter2D(img,-1, kernel)
    df_img_res['Laplacian'] = np.reshape(image_test, (1, img.shape[0]*img.shape[1])).squeeze()

    #Filtro Bilateral
    image_test = cv2.bilateralFilter(img, d_bilateral, sigmaColor_bilateral, sigmaSpace_bilateral, cv2.BORDER_DEFAULT)
    df_img_res['Bilateral'] = np.reshape(image_test, (1, img.shape[0]*img.shape[1])).squeeze()

    return df_img_res

def labels_iForest(attr, shape_img_output, grade):
    """
    La función aplicaráa Isolation Forest a un conjunto de datos de atributos de una imagen,
    posteriormente etiquetará cada vector dependiendo del grado que sea ingresado.
    Entrada:
        attr: Conjunto de datos con atributos de una imagen.
        shape_img_output: Especifica las dimensiones de salida de la imagen.
        grade: núúmero de etiquetas que se aplicarán al conjunto.
    Salida:
        labels: Etiquetas para cada vector.
    """
    # Se selecciona la cantidad de árboles y máximo tamaño de los árboles,
    # según la publicación original.
    clf = IsolationForest(n_estimators=100, max_samples=256)
    clf.fit(attr)
    outlier_score_img = np.floor(-grade * clf.score_samples(attr))
    return np.reshape(outlier_score_img, shape_img_output)

for i in range(1, 101):
    train_tmp  = io.imread("C:\\Users\\joy_p\\Desktop\\Proyecto_comia_2021\\version2\\Train\\tr_im{}.png".format(i))
    lung_mask_tmp = io.imread("C:\\Users\\joy_p\\Desktop\\Proyecto_comia_2021\\version2\\LungMask\\tr_lungmask{}.png".format(i))
    train_lung_mask_tmp = pixRelevant(train_tmp, lung_mask_tmp)
    df_tmp = createAttrImg(train_lung_mask_tmp)
    df_tmp.to_csv("C:\\Users\\joy_p\\Desktop\\Proyecto_comia_2021\\version2\\Attr\\attr_img{}.csv".format(i), sep='\t', index=None)

    mask_exp_tmp = io.imread("C:\\Users\\joy_p\\Desktop\\Proyecto_comia_2021\\version2\\Mask\\tr_mask{}.png".format(i))
    labels_mask_tmp = numLabels(mask_exp_tmp)
#    df_tmp = pd.read_csv("/content/drive/MyDrive/comia2021/Attr/attr_img{}.csv".format(i), sep='\t')
    labels_tmp = labels_iForest(df_tmp, (512, 512), labels_mask_tmp)
    df_out = pd.DataFrame(np.reshape(labels_tmp, (262144, 1)), columns=(['Label']), index=None)
    df_out.to_csv("C:\\Users\\joy_p\\Desktop\\Proyecto_comia_2021\\version2\\Result_Data\\res_data{}.csv".format(i), sep='\t')
    io.imsave("C:\\Users\\joy_p\\Desktop\\Proyecto_comia_2021\\version2\\Result_Image\\res_img{}.png".format(i), labels_tmp)
