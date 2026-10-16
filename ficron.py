import numpy as np
import torch
import cv2
from pytorch_models import UNet, UNet_Parasitos

from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter

#Basicamente, ignorar un poco lo que ya esta aqui, y obretodo usar el que esta en el codigo de la fase3

# Configurar dispositivo (GPU si está disponible, sino CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

#%%


"""
COUNT CELLS DE FICRON


def count_cells(img_org, return_map=False):

    size_img = np.array(img_org.shape[:2])

    size_img -= (size_img % 32)
    img_org = cv2.resize(img_org, (size_img[1], size_img[0]), interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img_org, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
    img1 = img1 / 65535.
    resul_predict = np.zeros((3,))

    #Prediccion parasitos
    with torch.no_grad():
        img_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        mask_p = model_p(img_tensor).squeeze().cpu().numpy()
    
    mask_p = mask_p / 3400.
    resul_predict[0] = mask_p.sum()
    mask_p = cv2.resize(mask_p, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA) * 4
    mask_p = mask_p * 3400.

    img1 = cv2.resize(img1, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)

    #Prediccion macrofagos
    with torch.no_grad():
        img_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        mask_c = model_c(img_tensor).squeeze().cpu().numpy()
    
    resul_predict[1] = mask_c.sum() / 1031.

    img_aux = np.concatenate((np.expand_dims(img1[:, :, 0], axis=-1), np.expand_dims(mask_p, axis=-1),
                              np.expand_dims(mask_c, axis=-1)), axis=-1)

    #Prediccion infectados con concatenacion de parasitos y macrofagos
    with torch.no_grad():
        img_tensor = torch.from_numpy(img_aux).permute(2, 0, 1).unsqueeze(0).float().to(device)
        mask_i = model_i(img_tensor).squeeze().cpu().numpy()
    
    thresh_inf = 0.8
    mask_i[mask_i >= thresh_inf] = 1
    mask_i[mask_i < (1 - thresh_inf)] = 0
    mask_i = mask_i * mask_c

    resul_predict[2] = mask_i.sum() / 1031.

    if return_map:
        return [mask_p, mask_c, mask_i], resul_predict
    else:
        return resul_predict
"""



#Solo se va a procesar una img a la vez, como en ficron
#Conteo de celulas y parasitos, con mapas de densidad del TFM de Miguel
def count_cells(img_org, channel=None, return_map=False):
    """
    Procesa la imagen y realiza predicciones según el canal especificado.
    
    Args:
        img_org: Imagen de entrada
        channel: Canal a procesar (0 para CH1/núcleos, 2 para CH3/parásitos, None para ambos) (Aunque no deberia hacerse nunca lo de ambos)
        return_map: Si True, devuelve los mapas de densidad
    
    Returns:
        Si return_map=True: ([mask_p, mask_c, mask_i], resul_predict)
        Si return_map=False: resul_predict
    """
    #PROCESADO IMG COMO EN TFM
    #Se asume que img está cargada
    if img_org is None:
        raise ValueError("img_org no puede ser None")

    #Convertir a numpy y a rgb si es posible
    #Transpose para que el canal esté en la primera dimensión (C, H, W) como espera PyTorch
    if hasattr(img_org, 'convert'):
        img_rgb = img_org.convert("RGB")
        image_np = np.array(img_rgb).transpose(2, 0, 1)
    else:
        image_np = np.array(img_org).transpose(2, 0, 1)

    #Se pasa a tensor
    img_tensor = torch.FloatTensor(image_np).unsqueeze(0)

    # Inicializar máscaras vacías
    mask_p = None
    mask_c = None

    # Validar canal si se especifica (parásitos o núcleos (CH1, CH3))
    if channel is not None and channel not in [0, 2]:
        raise ValueError("channel debe ser 0, 2 o None")

    selected_model_p = model_p
    selected_model_c = model_c

    # Predicción según el canal (parásitos o núcleos)
    if channel is None or channel == 2:
        # Canal CH3 (parásitos)
        with torch.no_grad():
            mask_p = selected_model_p(img_tensor).squeeze().detach().cpu().numpy()
    
    if channel is None or channel == 0:
        # Canal CH1 (núcleos)
        with torch.no_grad():
            mask_c = selected_model_c(img_tensor).squeeze().detach().cpu().numpy()

    # Crear máscaras vacías para los canales no procesados
    if mask_p is None:
        h, w = image_np.shape[1], image_np.shape[2]
        mask_p = np.zeros((h, w), dtype=np.float32)
    if mask_c is None:
        h, w = image_np.shape[1], image_np.shape[2]
        mask_c = np.zeros((h, w), dtype=np.float32)

    # Máscara de infectados vacía porque se esperan 3 valores de mascara
    mask_i = np.zeros_like(mask_c, dtype=np.float32)

    resul_predict = np.zeros((3,), dtype=float)
    resul_predict[0] = float(mask_p.sum())
    resul_predict[1] = float(mask_c.sum())
    resul_predict[2] = 0.0

    if return_map:
        return [mask_p, mask_c, mask_i], resul_predict

    return resul_predict


def count_cells_batch(tiles, channel=None, batch_size=2):
    """Ejecuta count_cells por lotes y devuelve un mapa por cada tesela."""
    if channel is not None and channel not in [0, 2]:
        raise ValueError("channel debe ser 0, 2 o None")
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero")

    maps = []
    for start in range(0, len(tiles), batch_size):
        batch_tiles = [np.asarray(tile) for tile in tiles[start:start + batch_size]]
        batch = np.stack([
            tile if tile.ndim == 3 else np.repeat(tile[..., np.newaxis], 3, axis=2)
            for tile in batch_tiles
        ])
        batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)

        with torch.no_grad():
            predictions_p = (
                model_p(batch_tensor).detach().cpu().numpy()
                if channel in (None, 2) else None
            )
            predictions_c = (
                model_c(batch_tensor).detach().cpu().numpy()
                if channel in (None, 0) else None
            )

        for index in range(len(batch_tiles)):
            prediction_p = (
                np.squeeze(predictions_p[index])
                if predictions_p is not None else None
            )
            prediction_c = (
                np.squeeze(predictions_c[index])
                if predictions_c is not None else None
            )
            shape = prediction_p.shape if prediction_p is not None else prediction_c.shape
            empty = np.zeros(shape, dtype=np.float32)
            maps.append(
                [
                    prediction_p if prediction_p is not None else empty,
                    prediction_c if prediction_c is not None else empty,
                    empty.copy(),
                ]
            )

    return maps





#Esta funcion va a hacer a prediccion sobre todas las teselas
#Todas las teselas deberian formar la imagen original completa. Si no se ha teselado porque
#img original ya era de 1024x1024, entonces solo habra una 'tesela'

#A diferencia de count_cells, esta funcion no se llamara dos veces (una para parasitos y otra para nucleos), sino que va a hacer primero la prediccion de parasitos sobre todas las teselas, y despues lo mismo pero con nucleos

#Va a realizar primero la prediccion de parasitos sobre todas las teselas, y despues lo mismo pero con nucleos
def predicciones_tfg(teselas):

    #mapas de densidad predichos de las teselas
    teselas_pred_parasitos = []
    teselas_pred_nucleos = []


    #Modelo de los parasitos (model_p) que se asume ya cargado
    model_p.eval()

    with torch.no_grad():
        for tile in teselas:
            # Convertir a Tensor PyTorch: (H, W, C) -> (1, C, H, W)
            tensor_tile = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor_tile = tensor_tile.to(device)

            # Inferencia
            pred_tile = model_p(tensor_tile)

            # Guardar resultado como numpy array 2D
            pred_np = pred_tile.squeeze().cpu().numpy()
            teselas_pred_parasitos.append(pred_np)  #Se guardan predicciones


    #Modelo de los nucleos celulares (model_c) que se asume ya cargado
    model_c.eval()

    with torch.no_grad():
        for tile in teselas:
            # Convertir a Tensor PyTorch: (H, W, C) -> (1, C, H, W)
            tensor_tile = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor_tile = tensor_tile.to(device)

            # Inferencia
            pred_tile = model_c(tensor_tile)

            # Guardar resultado como numpy array 2D
            pred_np = pred_tile.squeeze().cpu().numpy()
            teselas_pred_nucleos.append(pred_np)  #Se guardan predicciones


    return teselas_pred_parasitos, teselas_pred_nucleos #Se devuelven en este caso los dos tipos de mapas de densidad redichos para cada cosa














#Para cargar el modelo pytorch
def load_model(path, model_class):
    """
    Carga un modelo PyTorch desde un archivo .pth
    
    Args:
        path: Ruta al archivo del modelo .pth
        model_class: Clase del modelo PyTorch
    
    Returns:
        Modelo cargado en modo evaluación
    """
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modelo cargado: {path}")
    return model



















#La fase 3 no tiene modelo propio, si no que asigna los parasitos encontrados en la fase 1 con las celulas de la fase 2

# Rutas de los modelos PyTorch de Miguel (TFM)
path_model_p = 'Models/TFM_Miguel/fase1_modelo8.pth' 
path_model_cel = 'Models/TFM_Miguel/fase2_modelo2.pth'

# Rutas de los modelos PyTorch de mi TFG
path_model_p_tfg = 'Models/TFG_Lydia/Parasitos.pth'
path_model_cel_tfg = 'Models/TFG_Lydia/nucleos2.pth'


# Cargar modelos PyTorch
# NOTA: Ajusta las clases de modelos según la arquitectura de cada uno

#Modelos utilizados por TFM (trabajo anterior) por defecto
#model_p = load_model(path_model_p, model_class=UNet)
#model_c = load_model(path_model_cel, model_class=UNet)

model_p = None
model_c = None

#Modelos usados en mi TFG
#model_p_tfg = None
#model_cel_tfg = None
_detection_mode = 1


def set_detection_mode(mode):
    """Selecciona los modelos que se utilizaran en el siguiente conteo."""
    global _detection_mode, model_p, model_c

    #Si el error no es ni 1 ni 2
    if mode not in (1, 2):
        raise ValueError("mode debe ser 1 o 2")

    #Modelos del TFM de MIguel
    if mode == 1:
        print("Usando modelos de TFM (fase 1 y fase 2)")
        # Cargar modelos de TFM
        model_p = load_model(path_model_p, model_class=UNet)        
        model_c = load_model(path_model_cel, model_class=UNet)

    #Mis modelos del TFG
    elif mode == 2:
        model_p = load_model(path_model_p_tfg, model_class=UNet_Parasitos)
        model_c = load_model(path_model_cel_tfg, model_class=UNet)

    _detection_mode = mode
