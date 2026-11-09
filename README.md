## Interfaz de detección y conteo de parásitos en imágenes microscópicas

La interfaz permite cargar imágenes de fluorescencia, analizarlas para detectar células y parásitos y realizar la asignación entre ellos para determinar la carga parasitaria de la imagen total. Reconoce imágenes multicanal `CH1`–`CH4`, permite seleccionar el canal de visualización y realiza la detección y conteo de estructuras celulares descrito en el TFG de la propietaria de este repositorio.

Muestra mapas de densidad, conteos, asociaciones entre células y parásitos, carga parasitaria y gráficos de resultados, con zoom y visualización en blanco y negro.

Esta interfaz ha sido desarrollada como parte del Trabajo de Fin de Grado de la propietaria de este repositorio.


## Origen y cita

La interfaz se basa en el método FiCRoN y parte del repositorio original:

https://github.com/JoseMPerezVictoriaLab/FiCRoN

Si se utiliza FiCRoN, debe citarse el siguiente artículo:

Juez-Castillo, G., Valencia-Vidal, B., Orrego, L. M., Cabello-Donayre, M., Montosa-Hidalgo, L., & Pérez-Victoria, J. M. (2023). FiCRoN, a deep learning-based algorithm for the automatic determination of intracellular parasite burden from fluorescence microscopy images. Medical Image Analysis, 103036. [Paper](https://doi.org/10.1016/j.media.2023.103036)

## Dependencias

Se necesita Python 3 y las siguientes bibliotecas:

- PyQt5
- NumPy
- PyTorch y torchvision
- Pillow
- OpenCV (`opencv-python`)
- scikit-learn
- SciPy
- scikit-image
- seaborn
- matplotlib
- flowdec (para el módulo de deconvolución)

También deben estar disponibles los modelos entrenados y las carpetas de datos utilizadas por la aplicación.

# FiCRoN
FiCRoN. A novel method based on fully convolutional regression networks (FCRNs), for estimating intracellular parasite burden. This estimation requires three values, intracellular parasites, infected cells and uninfected cells.

## CITATION
If you use FiCRoN, please cite [paper](https://doi.org/10.1016/j.media.2023.103036):

Juez-Castillo, G., Valencia-Vidal, B., Orrego, L. M., Cabello-Donayre, M., Montosa-Hidalgo, L., & Pérez-Victoria, J. M. (2023). FiCRoN, a deep learning-based algorithm for the automatic determination of intracellular parasite burden from fluorescence microscopy images. Medical Image Analysis, 103036.

![ART_BMDM](https://github.com/JoseMPerezVictoriaLab/FiCRoN/assets/45313868/073f04c6-d697-4639-a37d-930d8adda0ae)
