#%%
from PIL import Image, ImageSequence
import numpy as np
from read_lif import Reader as r_lif
import ficron
import cv2

#TFM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import ImageDraw
import pandas as pd
import os
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from collections import defaultdict



def read_tiff(path, size_img=None):
    """
    path - Path to the multipage-tiff file
    """
    images = []

    with Image.open(path) as img:
        for im in ImageSequence.Iterator(img):
            if size_img is None:
                images.append(np.array(im))
            else:
                images.append(np.array(im.resize((size_img, size_img))))

    img_temp = np.stack(images, axis=-1)

    return img_temp


def read_lif(path):
    series_img = r_lif.Reader('/'.join([dirs, file]))
    for choose_img in series_img:
        yield np.stack([choose_img.getFrame2D(T=0, channel=0, dtype=np.uint16),
                          choose_img.getFrame2D(T=0, channel=1, dtype=np.uint16)], axis=-1)


def croll_image(full_image, idx, image_size):
    # print(full_image.shape, idx)
    return full_image[idx[0]: idx[0] + image_size,
            idx[1]: idx[1] + image_size]


def save_multi_frame_tiff(image_array, size_img, name_file):

    list_img = [Image.fromarray(image_array[:, :, idx])
            for idx in range(image_array.shape[-1])]
    list_img[0].save(name_file, save_all=True,
                     append_images=list_img[1:])

def convert_uint16(img):
    imin = img.min(axis=(0, 1))
    imax = img.max(axis=(0, 1))
    
    # Evitar división por cero cuando imax == imin (imagen uniforme)
    denom = imax - imin
    denom = np.where(denom == 0, 1, denom)

    return (65535 * (img - imin) / denom).astype(np.uint16)

def convert_uint8(img):
    imin = img.min(axis=(0, 1))
    imax = img.max(axis=(0, 1))
    
    # Evitar división por cero cuando imax == imin (imagen uniforme)
    denom = imax - imin
    denom = np.where(denom == 0, 1, denom)
    
    return (255. * (img - imin) / denom).astype(np.uint8)


def convert_gray2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def deconv_img(n_dcv):
    import deconv as dconv

    img = np.asarray(n_dcv)
    img = np.squeeze(img)

    if img.ndim == 2:
        img = np.stack([img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[-1] >= 2:
            img = img[:, :, :2]
        elif img.shape[0] >= 2:
            img = np.moveaxis(img[:2, :, :], 0, -1)
        else:
            raise ValueError(f"No se pudieron obtener 2 canales de la imagen: {img.shape}")
    else:
        raise ValueError(f"Formato de imagen no soportado para deconvolución: {img.shape}")

    n_dcv_transf = np.moveaxis(img, -1, 0)
    dcv_img = dconv.decv_img(n_dcv_transf, dcv_kernel)
    dcv_img = convert_uint16(dcv_img)
    return dcv_img


def count_img(img_org, return_map=False, dcv=False):
    if dcv:
        img_org = deconv_img(img_org)
    return (img_org,) + ficron.count_cells(img_org, return_map=return_map)


def merge_img_map(img_org, maps):
    merge_img = img_org.copy()
    size_img = merge_img.shape[:2]

    tmp_cellmap = cv2.resize(maps[1] * 22, size_img, interpolation=cv2.INTER_NEAREST)
    mask_tmp = tmp_cellmap > 150
    merge_img[mask_tmp] = [255, 255, 0]

    tmp_cellmap = cv2.resize(maps[2] * 22, size_img, interpolation=cv2.INTER_NEAREST)
    mask_tmp = tmp_cellmap > 150
    merge_img[mask_tmp] = [255, 0, 0]

    merge_img[:, :, 2] = merge_img[:, :, 2] + cv2.resize(maps[0] * 2, size_img, interpolation=cv2.INTER_NEAREST)

    return merge_img


psf_dapi = read_tiff('PSF/PSF_DAPI_63x.tif')
psf_cellMask = read_tiff('PSF/PSF_cellmask_63x.tif')
dcv_kernel = np.concatenate([psf_cellMask, psf_dapi], axis=-1)
dcv_kernel = np.moveaxis(dcv_kernel, [0, 1, 2], [1, 2, 0])










#A PARTIR DE AQUI SE VAN A AÑADIR LAS FUNCIONES DE CONTABILIZACIÓN DE PARÁSITOS ETC QUE SE TIENEN DEL CÓDIGO DEL TFM


#PARASITOS

def contar_parasitos_dbscan(img_original, density_map, threshold_rel=0.35, eps=2, min_samples=2):
    density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min())
    threshold = threshold_rel * np.max(density_map)
    y_coords, x_coords = np.where(density_map > threshold)
    coords = np.column_stack((x_coords, y_coords))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    centros = [np.mean(coords[labels == label], axis=0) for label in unique_labels]
    return len(centros), centros

def evaluar_parasitos(df_preds, carpeta_imagenes_test_final, nums_parasitos):
    carpeta_salida = os.path.join(carpeta_imagenes_test_final, "Predichos")
    carpeta_coords = os.path.join(carpeta_salida, "coordenadas_predichas")
    os.makedirs(carpeta_coords, exist_ok=True)

    resultados = []
    for i, row in df_preds.iterrows():
        nombre = row["nombre"]
        pred = row["pred"]
        path_original = carpeta_imagenes_test_final + f"/{nombre}_etiquetada.png"
        img_original = Image.open(path_original)
        img_original = np.array(img_original)
        nombre_img = nombre + ".tif"
        real = nums_parasitos[nums_parasitos.Imagen == nombre_img]["Número de Puntos"].values[0]

        if np.max(pred) < 20:
            num_dbscan, centros_dbscan = 0, []
        else:
            num_dbscan, centros_dbscan = contar_parasitos_dbscan(img_original, pred)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_original)
        axes[0].set_title(f"Original: {real}")
        axes[0].axis('off')

        axes[1].imshow(pred, cmap='gray')
        for c in centros_dbscan:
            axes[1].plot(c[0], c[1], 'ro', markersize=2)
        axes[1].set_title(f"DBSCAN: {num_dbscan}")
        axes[1].axis('off')

        plt.suptitle(nombre)
        plt.tight_layout()
        plt.show()

        print(nombre)
        print("Real:", real)
        print("Predicho DBSCAN:", num_dbscan)
        print("------")

        resultados.append({
            "Nombre": nombre,
            "Real": real,
            "Predicho_DBSCAN": num_dbscan
        })

        path_ch4 = os.path.join(carpeta_salida, f"{nombre[:-4]}_CH4_pred.png")
        if os.path.exists(path_ch4):
            img_ch4 = Image.open(path_ch4).convert("RGB")
            draw = ImageDraw.Draw(img_ch4)
            path_coords = os.path.join(carpeta_coords, f"{nombre[:-4]}_CH4_pred.txt")
            with open(path_coords, "w") as f:
                for c in centros_dbscan:
                    f.write(f"{c[0]},{c[1]}\n")
                    x, y = int(round(c[0])), int(round(c[1]))
                    draw.line((x-3, y, x+3, y), fill="yellow", width=1)
                    draw.line((x, y-3, x, y+3), fill="yellow", width=1)
            img_ch4.save(path_ch4)

    return pd.DataFrame(resultados)


#Esto es para que solo muestre las predicciones sobre la img original, sin mostrar las etiquetas reales
def evaluar_parasitos_solo_pred(df_preds, carpeta_imagenes_test_final):
    carpeta_salida = os.path.join(carpeta_imagenes_test_final, "Predichos")
    carpeta_coords = os.path.join(carpeta_salida, "coordenadas_predichas")
    os.makedirs(carpeta_coords, exist_ok=True)

    resultados = []
    for _, row in df_preds.iterrows():
        nombre = row["nombre"]
        pred = row["pred"]

        # Misma ruta de imagen que en evaluar_parasitos
        path_original = os.path.join(carpeta_imagenes_test_final, f"{nombre}.png")
        img_original = np.array(Image.open(path_original))

        if np.max(pred) < 20:
            num_dbscan, centros_dbscan = 0, []
        else:
            num_dbscan, centros_dbscan = contar_parasitos_dbscan(img_original, pred)

        # Mostrar solo predicción sobre la imagen
        plt.figure(figsize=(6, 6))
        plt.imshow(img_original)
        for c in centros_dbscan:
            plt.plot(c[0], c[1], 'yo', markersize=3)
        plt.title(f"{nombre} - Predicción DBSCAN: {num_dbscan}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        print(nombre)
        print("Predicho DBSCAN:", num_dbscan)
        print("------")

        resultados.append({
            "Nombre": nombre,
            "Predicho_DBSCAN": num_dbscan
        })

        # Guardado de coordenadas e imagen CH4 como en la función original
        path_ch4 = os.path.join(carpeta_salida, f"{nombre[:-4]}_CH4_pred.png")
        if os.path.exists(path_ch4):
            img_ch4 = Image.open(path_ch4).convert("RGB")
            draw = ImageDraw.Draw(img_ch4)
            path_coords = os.path.join(carpeta_coords, f"{nombre[:-4]}_CH4_pred.txt")
            with open(path_coords, "w") as f:
                for c in centros_dbscan:
                    f.write(f"{c[0]},{c[1]}\n")
                    x, y = int(round(c[0])), int(round(c[1]))
                    draw.line((x-3, y, x+3, y), fill="yellow", width=1)
                    draw.line((x, y-3, x, y+3), fill="yellow", width=1)
            img_ch4.save(path_ch4)

    return pd.DataFrame(resultados)



#NUCLEOS CELULARES

def detectar_parasitos_maxima_parametrizable(pred, sigma_suavizado=3, min_distance=30, percentil_fondo=80):
    smoothed = gaussian_filter(pred, sigma=sigma_suavizado)
    coordinates = peak_local_max(smoothed, min_distance=min_distance)

    background_threshold = np.percentile(smoothed, percentil_fondo)

    filtered_coords = []
    for y, x in coordinates:
        val_centro = smoothed[y, x]
        vecindario = smoothed[max(y-3,0):y+4, max(x-3,0):x+4]
        val_medio_around = np.mean(np.delete(vecindario.flatten(), vecindario.size // 2))
        if val_centro > background_threshold and val_centro - val_medio_around > 0.05:
            filtered_coords.append((y, x))

    return len(filtered_coords), filtered_coords

def evaluar_modelo(df_preds, carpeta_imagenes_test, carpeta_imagenes_test_final, nums, sigma_suavizado=5, min_distance=21, percentil_fondo=20):
    carpeta_salida = os.path.join(carpeta_imagenes_test_final, "Predichos")
    os.makedirs(carpeta_salida, exist_ok=True)

    for nombre in df_preds["nombre"].unique():
        real = nums[nums.Imagen == nombre + ".tif"]["Número de Puntos"].values[0]
        img_original = np.array(Image.open(f"{carpeta_imagenes_test_final}/{nombre}_etiquetada.png"))
        pred = df_preds[(df_preds['nombre'] == nombre)]["pred"].values[0]
        
        n_detected, centros = detectar_parasitos_maxima_parametrizable(
            pred, sigma_suavizado, min_distance, percentil_fondo
        )

        # Mostrar resultados como antes
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(img_original)
        for c in centros:
            axes[0].plot(c[1], c[0], 'r+', markersize=3, markeredgewidth=3)
        axes[0].set_title(f"Original: {real} / Detectados: {n_detected}")
        axes[0].axis('off')

        axes[1].imshow(pred, cmap='gray')
        for c in centros:
            axes[1].plot(c[1], c[0], 'ro', markersize=2)
        axes[1].set_title("Predicción modelo2")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        # Cargar imagen CH4
        path_ch4 = os.path.join(carpeta_imagenes_test, f"{nombre[:-4]}_CH4.tif")
        img_ch4 = Image.open(path_ch4).convert("RGB")
        draw = ImageDraw.Draw(img_ch4)
        path_coords = os.path.join(carpeta_imagenes_test_final,"Predichos/coordenadas_predichas", f"{nombre[:-4]}_CH1_pred.txt")
        with open(path_coords, "w") as f:
            for c in centros:
                f.write(f"{c[1]},{c[0]}\n")
                x, y = c[1], c[0]
                draw.ellipse((x-2, y-2, x+2, y+2), outline="red", width=1)
        img_ch4.save(os.path.join(carpeta_salida, f"{nombre[:-4]}_CH4_pred.png"))


#Hace lo mismo que evaluar_modelo pero sin mostrar la imagen original ni el número real de parásitos, solo mostrando las predicciones sobre la imagen original
def evaluar_modelo_solo_pred_1fig(
    df_preds,
    carpeta_imagenes_test,
    carpeta_imagenes_test_final,
    sigma_suavizado=5,
    min_distance=21,
    percentil_fondo=20
):
    carpeta_salida = os.path.join(carpeta_imagenes_test_final, "Predichos")
    carpeta_coords = os.path.join(carpeta_salida, "coordenadas_predichas")
    os.makedirs(carpeta_salida, exist_ok=True)
    os.makedirs(carpeta_coords, exist_ok=True)

    for nombre in df_preds["nombre"].unique():
        base_nombre = nombre[:-4] if str(nombre).lower().endswith(".tif") else str(nombre)

        # Imagen original sin etiqueta real
        path_img_original = os.path.join(carpeta_imagenes_test_final, f"{base_nombre}.png")
        img_original = np.array(Image.open(path_img_original))

        pred = df_preds[df_preds["nombre"] == nombre]["pred"].values[0]
        n_detected, centros = detectar_parasitos_maxima_parametrizable(
            pred, sigma_suavizado, min_distance, percentil_fondo
        )

        # Solo 1 figura: original + detecciones
        plt.figure(figsize=(6, 6))
        plt.imshow(img_original)
        for c in centros:
            plt.plot(c[1], c[0], "r+", markersize=4, markeredgewidth=1.5)
        plt.title(f"{base_nombre} - Detectados: {n_detected}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        print(base_nombre)
        print("Detectados:", n_detected)
        print("------")

        # Guardado de CH4 + coordenadas
        path_ch4 = os.path.join(carpeta_imagenes_test, f"{base_nombre}_CH4.tif")
        if os.path.exists(path_ch4):
            img_ch4 = Image.open(path_ch4).convert("RGB")
            draw = ImageDraw.Draw(img_ch4)

            path_coords = os.path.join(carpeta_coords, f"{base_nombre}_CH1_pred.txt")
            with open(path_coords, "w") as f:
                for c in centros:
                    x, y = float(c[1]), float(c[0])
                    f.write(f"{x},{y}\n")
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="red", width=1)

            img_ch4.save(os.path.join(carpeta_salida, f"{base_nombre}_CH4_pred.png"))



#ASIGNACIÓN PARÁSITO-NUCELO CELULAR

def asociar_por_clustering(coords_ch1, coords_ch4, draw, eps=25, min_samples=1):
    asociaciones = defaultdict(list)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_ch4)
    labels = clustering.labels_
    for label in set(labels):
        if label == -1:
            continue  # ruido
        cluster = coords_ch4[labels == label]
        centroide = np.mean(cluster, axis=0)
        distancias = np.linalg.norm(coords_ch1 - centroide, axis=1)
        idx_min = np.argmin(distancias)
        for par in cluster:
            asociaciones[idx_min].append(par)
            x1, y1 = par
            x2, y2 = coords_ch1[idx_min]
            draw.line((x1, y1, x2, y2), fill="cyan", width=1)
    return asociaciones

def asociar_por_distancia_minima(coords_ch1, coords_ch4, draw):
    asociaciones = defaultdict(list)
    for par in coords_ch4:
        distancias = np.linalg.norm(coords_ch1 - par, axis=1)
        idx_min = np.argmin(distancias)
        asociaciones[idx_min].append(par)
        x1, y1 = par
        x2, y2 = coords_ch1[idx_min]
        draw.line((x1, y1, x2, y2), fill="cyan", width=1)
    return asociaciones

def resumen(asociaciones, total_celulas):
    infectadas = sum(1 for v in asociaciones.values() if len(v) > 0)
    sanas = total_celulas - infectadas
    total_parasitos = sum(len(v) for v in asociaciones.values())
    media_parasitos = total_parasitos / infectadas if infectadas > 0 else 0
    #return infectadas, sanas, media_parasitos

    #Se ha modificado esta funcion del TFM para que además calcule el porcentaje de células infectadas
    pct_infectadas = (infectadas / total_celulas * 100) if total_celulas > 0 else 0
    return infectadas, sanas, media_parasitos, pct_infectadas

def analizar_infeccion(nombre, carpeta_coords, carpeta_salida):
    path_ch1 = os.path.join(carpeta_coords, f"{nombre}_CH1_pred.txt")
    path_ch4 = os.path.join(carpeta_coords, f"{nombre}_CH4_pred.txt")
    coords_ch1 = np.loadtxt(path_ch1, delimiter=",")
    coords_ch4 = np.loadtxt(path_ch4, delimiter=",")

    if coords_ch1.ndim == 1:
        coords_ch1 = np.array([coords_ch1])
    if coords_ch4.ndim == 1:
        coords_ch4 = np.array([coords_ch4])

    print(f"{nombre}:")
    print(f"Número de células (núcleos): {len(coords_ch1)}")
    print(f"Número de parásitos: {len(coords_ch4)}")

    path_img = os.path.join(carpeta_salida, f"{nombre}_CH4_pred.png")
    img_base = Image.open(path_img).convert("RGB")

    # Versión 1: distancia mínima
    img1 = img_base.copy()
    draw1 = ImageDraw.Draw(img1)
    asociaciones1 = asociar_por_distancia_minima(coords_ch1, coords_ch4, draw1)
    inf1, san1, media1, pct1 = resumen(asociaciones1, len(coords_ch1))

    # Versión 2: clustering
    img2 = img_base.copy()
    draw2 = ImageDraw.Draw(img2)
    asociaciones2 = asociar_por_clustering(coords_ch1, coords_ch4, draw2)
    inf2, san2, media2, pct2 = resumen(asociaciones2, len(coords_ch1))

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(img1)
    axs[0].set_title(f"{nombre} - Distancia mínima")
    axs[0].axis("off")
    axs[1].imshow(img2)
    axs[1].set_title(f"{nombre} - Clustering + centroide")
    axs[1].axis("off")
    plt.show()

    print(f"[Distancia mínima] Células infectadas: {inf1}, sanas: {san1}, media parásitos: {media1:.2f}")
    print(f"[Clustering] Células infectadas: {inf2}, sanas: {san2}, media parásitos: {media2:.2f}")