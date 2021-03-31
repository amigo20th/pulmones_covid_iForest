# pulmones_covid_iForest
Segmentación por medio de isolation forest en imágenes de pulmones dañados por SARS-CoV-2.

# Información sobre el proyecto.

Se va segmentar las secciones dañadas por COVID-19 en pulmones, para eso se cuentan con 100 imágenes de pulmones revisadas por un ojo experto.

Las 100 imágenes originales obtenidas por medio de un dispositivo médico (tomografía por emisión de positrones) se encuentran en el conjunto **Train.zip**.

Para seleccionar la información relevante a nuestro estudio, es necesario seccionar la imagen del pulmón de lo que no lo es, para eso se cuenta con el conjunto de imagenes llamado **LungMask.zip**.

Las observaciones hechas por un ojo experto, es decir, las áreas dañadas que fueron seleccionadas por un doctor se encuentran en el conjunto de imágenes **Mask.zip**.

# El código solo es ejecutable en python 2
# Para correr el código por bash
En el presente código solo se va a seccionar las imágenes por medio de _isolation forest_, por esa razón, y para facilitar las ejecuciones, se creo un script en bash.

En nombre del script es **extract_if.sh**, en el cual son necesarios 2 parámetros: _i_ es el número de imagen a aplicar el algoritmo de _isolation forest_, el otro parámetro es _k_, el cual especifica el número de clases que se va a segmentar la imagen.
Ejemplo de ejecución: 
**extract_if.sh -i 1 -6**

El anterior script estaría dando como resultado la segmentación en 6 clases de la imagen 1.

# Para ejecución por código de _Python_.
Para poder ejecutar en el código que contiene el script pero directamente en _Python_ se necesita ejecutar 2 archivos _.py_

1- El primer código _Python_ necesario es **extract_atts_lung.py**, es el encargado de aplicar la máscara, cada pixel es mapeado del espacio de dimensión 1 (su nivel de gris, de 0 a 255) a un espacio de dimensión _k_, donde _r_ es el número de sus vecinos. Cada pixel _p_ es rodeado por un kernel uniforme, centrado en _p_, de lado _k_.

El script tiene como parámetros de entrada: _i_, _m_ y _c_ que son las imágenes de entrada, también cuenta con _nk_ como parámetro de entrada, este se encarga de especiicar el número de clases a segmentar la imagen.

Este script tiene como salidas las imágenes que se encuentran en la carpeta **new_images**. El nombre de dichas imagenes se puede estipular por los parámetros _o_ y _o2_. También cuenta como salida los archivos _.csv_ los cuales contienen la información de los pixeles de las imágenes de salida, los nombre de estas salidas se puede especificar con _r_, _rr_ y _s_.

Ejemplo de ejecución para la imagen 1 y _k_=6:

**python2 extract_atts_lung.py  -i Train/tr_im1.png  -m  LungMask/tr_lungmask1.png  -c Mask/tr_mask1.png  -r results/im_1_k_6.csv  -rr results/im_1_k_6_n.csv"  -s results/im_1_k_6_pix.csv  -o new_images/im_1.png  -o2 new_images/im_1_k_6.png -nk 6**

2- El segundo archivo en _Python_ para realizar toda la segmentación es **AnomDet_Lung.py**, este es el encargado de aplicar _isolation forest_ a los datos de salida del paso anterior.

Los parámetros de entrada son el archivo _.csv_ resultante del paso anterior, _th_ es la cantidad de árboles de aislamiento que tendrá el bosque de aislamiento (_isolation forest_).

El parámetro de salida será el archivo _.csv_ de cada pixel y qué clase de la segmentación se le asignó.

Ejemplo e ejecución para la imagen 1 y _k_=6:

**python2 AnomDet_Lung.py  -i results/im_1_k_6_n.csv -m if  -th 10  -o AD/im_1_k_6_n_if_th_10.csv**




