import count_util as c_util
import ficron
import torch
import traceback

from GUI_Ficron import * # import PyQt5 widgets
from PyQt5.QtWidgets import QFileDialog
import PyQt5
import os
dirname = os.path.dirname(PyQt5.__file__)
plugin_path = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

from PIL import Image, ImageDraw
import numpy as np

#%%

def img_np2qt(img_np):
    img_rgb = np.ascontiguousarray(img_np)
    height, width, _ = img_rgb.shape
    qimg = QtGui.QImage(img_rgb.data, width, height, 3 * width, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.img_org_2ch = None
        self.img_org_rgb = None
        self.img_org_2ch_rgb = None
        self.density_maps = None
        self.img_view = None
        self.path_dir = None

        if hasattr(self, 'btn_analizar'):
            self.btn_analizar.clicked.connect(self.analizar_imagen_actual)


    def set_view_image(self, img_npRGB):
        view_w = max(1, self.view_image.width())
        view_h = max(1, self.view_image.height())
        img_view = img_np2qt(img_npRGB).scaled(view_w, view_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.view_image.setPixmap(img_view)

    def open_imag(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "Lote_final_procesado/", "Images (*.tif *.lif)",
                                                  options=options)
        if not fileName:
            self.state_process.setText('Selección cancelada.')
            return
        img_raw = np.squeeze(np.asarray(c_util.read_tiff(fileName)))
        if img_raw.ndim == 2:
            img_raw = np.stack([img_raw, img_raw, img_raw], axis=-1)
        elif img_raw.ndim == 3 and img_raw.shape[2] >= 3:
            img_raw = img_raw[:, :, :3]
        elif img_raw.ndim == 3 and img_raw.shape[0] >= 3:
            img_raw = np.moveaxis(img_raw[:3, :, :], 0, -1)
        else:
            raise ValueError(f"Formato de imagen no soportado: {img_raw.shape}")

        self.img_org_2ch = img_raw
        img_tmp = c_util.convert_uint8(self.img_org_2ch)
        self.img_org_rgb = img_tmp
        self.img_org_2ch_rgb = [c_util.convert_gray2bgr(img_tmp[:, :, i]) for i in range(3)]
        self.checkView_maps.setChecked(False)
        self.density_maps = None
        self.update_view()

    def result_count(self, data_result):
        self.res_par.setText(str(round(data_result[0], 1)))
        self.res_macrop.setText(str(round(data_result[1], 1)))
        self.res_inf.setText(str(round(data_result[2], 1)))


    #Esto llama al count_cells de ficron desde la funcion definida en count_util
    def count_img(self):
        self.analizar_imagen_actual()

    def update_view(self):
        if self.img_org_2ch is None:
            return
        if self.checkCell.isChecked():
            channel = 0
        else:
            channel = 2
        if self.checkView_maps.isChecked() and self.density_maps is not None:
            self.img_view = c_util.merge_img_map(self.img_org_2ch_rgb[channel], self.density_maps)
        else:
            self.img_view = self.img_org_rgb

        self.set_view_image(self.img_view)

    def select_path_dir(self):
        options = QFileDialog.Options()
        path_dir = QFileDialog.getExistingDirectory(self, "Open Image", "Lote_final_procesado/", QFileDialog.ShowDirsOnly)
        self.path_dir = path_dir
        self.text_path_dir.setText('.../' + self.path_dir.split('/')[-1])

    def count_dir(self):
        pass
        

    def _infer_maps(self):
        """
        Devuelve predicciones usando los modelos cargados de ficron.
        """

        """
        # Convertir imagen a tensor (3 canales: B nuclei, G membrane, R parasites)
        img_np = np.squeeze(np.asarray(self.img_org_2ch))
        if img_np.ndim == 2:
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
        elif img_np.ndim == 3 and img_np.shape[2] >= 3:
            img_np = img_np[:, :, :3]
        elif img_np.ndim == 3 and img_np.shape[0] >= 3:
            img_np = np.moveaxis(img_np[:3, :, :], 0, -1)
        else:
            raise ValueError(f"Formato de imagen no soportado para inferencia: {img_np.shape}")

        max_val = float(img_np.max()) if img_np.size else 0.0
        if max_val > 1.0:
            scale = 65535.0 if max_val > 255.0 else 255.0
        else:
            scale = 1.0

        img_ch1 = img_np[:, :, 0].astype(np.float32) / scale
        img_ch4 = img_np[:, :, 2].astype(np.float32) / scale

        # Stacking: (H, W, 3) - repetir CH1 para RGB fake si es necesario, o usar directamente
        # Asumo que el modelo espera RGB, así que replicamos CH1 y CH4
        img_rgb = np.stack([img_ch1, img_ch4, img_ch1], axis=-1)  # (H, W, 3)

        """

        #ESTO YA SE HACE CON COUNT_CELLS DE FICRON
        """
        # Convertir a tensor
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(ficron.device)

        # Predicción parásitos (CH4)
        with torch.no_grad():
            pred_ch4 = ficron.model_p(img_tensor).squeeze().cpu().numpy()

        # Predicción núcleos (CH1)
        with torch.no_grad():
            pred_ch1 = ficron.model_c(img_tensor).squeeze().cpu().numpy()
        """

# Llamar a count_cells de ficron.py para obtener las predicciones
        maps, resul_predict = ficron.count_cells(img_org=self.img_org_2ch, return_map=True)
        
        # maps = [mask_p, mask_c, mask_i]
        pred_ch1 = maps[1]  # mask_c (núcleos)
        pred_ch4 = maps[0]  # mask_p (parásitos)
        
        return pred_ch1, pred_ch4

    def analizar_imagen_actual(self):
        if self.img_org_2ch is None:
            self.state_process.setText("Primero carga una imagen.")
            return

        self.state_process.setText("Analizando...")

        # 1) Predicciones (núcleos/ parásitos)
        pred_ch1, pred_ch4 = self._infer_maps()

        # 2) Detección de núcleos (sale en formato (y, x))
        n_nucleos, centros_ch1_yx = c_util.detectar_parasitos_maxima_parametrizable(pred_ch1, sigma_suavizado=5, min_distance=21, percentil_fondo=20) #Se han puesto los valores que salían en evaluar modelo
        coords_ch1 = np.array([(float(x), float(y)) for (y, x) in centros_ch1_yx], dtype=float)

        # 3) Detección de parásitos (sale en formato (x, y))
        n_parasitos, centros_ch4_xy = c_util.contar_parasitos_dbscan( self.img_org_2ch_rgb[2], pred_ch4 )
        coords_ch4 = np.array(centros_ch4_xy, dtype=float) if len(centros_ch4_xy) else np.empty((0, 2), dtype=float)

        # 4) Overlay en imagen RGB completa
        base = self.img_org_rgb.copy()
        img_pil = Image.fromarray(base)
        draw = ImageDraw.Draw(img_pil)

        # Dibujar núcleos (rojo)
        for x, y in coords_ch1:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="red", width=1)

        # Dibujar parásitos (amarillo)
        for x, y in coords_ch4:
            draw.line((x - 3, y, x + 3, y), fill="yellow", width=1)
            draw.line((x, y - 3, x, y + 3), fill="yellow", width=1)

        # 5) Asignación parásito -> núcleo (líneas cian) SE HACE CON ASOCIACIÓN POR CLUSTERING PORQUE DABA MEJORES RESULTADOS
        if len(coords_ch1) > 0 and len(coords_ch4) > 0:
            asociaciones = c_util.asociar_por_clustering(coords_ch1, coords_ch4, draw)
            infectadas, sanas, media_parasitos = c_util.resumen(asociaciones, len(coords_ch1))
        else:
            asociaciones = {}
            infectadas, sanas, media_parasitos = 0, len(coords_ch1), 0.0

        # Mostrar en QLabel principal
        self.set_view_image(np.array(img_pil))

        # Reutiliza labels existentes (ajusta si tienes labels propios)
        self.res_par.setText(str(n_parasitos))          # parásitos detectados
        self.res_macrop.setText(str(n_nucleos))         # núcleos detectados
        self.res_inf.setText(str(infectadas))           # células infectadas

        self.state_process.setText(
            f"Listo | Núcleos: {n_nucleos}, Parásitos: {n_parasitos}, "
            f"Infectadas: {infectadas}, Sanas: {sanas}, Media: {media_parasitos:.2f}"
        )

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()