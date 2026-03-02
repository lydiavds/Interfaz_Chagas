import count_util as c_util
import ficron
import torch
import traceback
import re

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
        
        # Nuevas variables para manejo de canales separados
        self.img_ch1 = None  # Imagen del canal CH1 (núcleos)
        self.img_ch3 = None  # Imagen del canal CH3 (parásitos)
        self.r_number = None  # Número r compartido entre ambas imágenes
        self.current_filename = None
        self.channel_images = {}  # {1: np.array, 2: np.array, 3: np.array, 4: np.array}
        self.current_display_channel = 1

        self.coords_nucleos = np.empty((0, 2), dtype=float)
        self.coords_parasitos = np.empty((0, 2), dtype=float)
        self.association_lines = []  # [(x1, y1, x2, y2), ...]

        self.channel_group = QtWidgets.QButtonGroup(self)
        self.channel_group.setExclusive(True)
        self.channel_group.addButton(self.checkCell)
        #Se chequean ch2 y ch4 porque ch1 y ch3 se usan para la detección de parásitos y macrófagos
        if hasattr(self, 'checkCH2'):
            self.channel_group.addButton(self.checkCH2)
        self.channel_group.addButton(self.checkDapi)
        if hasattr(self, 'checkCH4'):
            self.channel_group.addButton(self.checkCH4)
        self.checkCell.setChecked(True)

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

        self.path_dir = os.path.dirname(fileName)
        self.text_path_dir.setText(fileName)
        
        # Extraer nombre del archivo sin extensión
        basename = os.path.basename(fileName)
        name_without_ext = os.path.splitext(basename)[0]
        dir_path = os.path.dirname(fileName)
        
        # Detectar patrón r*_CH* (cualquier canal)
        pattern = re.match(r'(r\d+)_CH(\d+)', name_without_ext)
        
        if pattern:
            # Imagen con patrón detectado
            self.r_number = pattern.group(1)
            current_channel = int(pattern.group(2))
            self.current_filename = fileName
            
            # Cargar la imagen actual (la que se mostró en el diálogo)
            img_raw = np.squeeze(np.asarray(c_util.read_tiff(fileName)))
            if img_raw.ndim == 2:
                img_raw = np.stack([img_raw, img_raw, img_raw], axis=-1)
            elif img_raw.ndim == 3 and img_raw.shape[2] >= 3:
                img_raw = img_raw[:, :, :3]
            elif img_raw.ndim == 3 and img_raw.shape[0] >= 3:
                img_raw = np.moveaxis(img_raw[:3, :, :], 0, -1)
            else:
                raise ValueError(f"Formato de imagen no soportado: {img_raw.shape}")
            
            self.channel_images = {}
            for channel_num in (1, 2, 3, 4):
                channel_path = os.path.join(dir_path, f"{self.r_number}_CH{channel_num}.tif")
                if os.path.exists(channel_path):
                    self.channel_images[channel_num] = self._load_single_image(channel_path)

            if current_channel not in self.channel_images:
                self.channel_images[current_channel] = img_raw

            self.img_ch1 = self.channel_images.get(1)
            self.img_ch3 = self.channel_images.get(3)

            if current_channel in self.channel_images:
                self.current_display_channel = current_channel
            elif 1 in self.channel_images:
                self.current_display_channel = 1
            elif 2 in self.channel_images:
                self.current_display_channel = 2
            elif 3 in self.channel_images:
                self.current_display_channel = 3
            elif 4 in self.channel_images:
                self.current_display_channel = 4

            self._set_channel_checkbox(self.current_display_channel)
            self._set_current_display_image()

            ch_available = ", ".join([f"CH{ch}" for ch in sorted(self.channel_images.keys())])
            self.state_process.setText(
                f"Cargado {self.r_number}_CH{current_channel} | Canales disponibles: {ch_available}"
            )
        else:
            # Imagen sin patrón, cargar de forma tradicional
            self.r_number = None
            self.img_ch1 = None
            self.img_ch3 = None
            self.channel_images = {}
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
            self.state_process.setText('Imagen cargada (modo tradicional)')

        self.coords_nucleos = np.empty((0, 2), dtype=float)
        self.coords_parasitos = np.empty((0, 2), dtype=float)
        self.association_lines = []
        
        self.checkView_maps.setChecked(False)
        self.density_maps = None
        self.update_view()
    
    def _load_single_image(self, filepath):
        """Carga una imagen individual en formato RGB numpy."""
        img_raw = np.squeeze(np.asarray(c_util.read_tiff(filepath)))
        if img_raw.ndim == 2:
            img_raw = np.stack([img_raw, img_raw, img_raw], axis=-1)
        elif img_raw.ndim == 3 and img_raw.shape[2] >= 3:
            img_raw = img_raw[:, :, :3]
        elif img_raw.ndim == 3 and img_raw.shape[0] >= 3:
            img_raw = np.moveaxis(img_raw[:3, :, :], 0, -1)
        else:
            raise ValueError(f"Formato de imagen no soportado: {img_raw.shape}")
        return img_raw

    def _set_channel_checkbox(self, channel_num):
        if channel_num == 1:
            self.checkCell.setChecked(True)
        elif channel_num == 2 and hasattr(self, 'checkCH2'):
            self.checkCH2.setChecked(True)
        elif channel_num == 3:
            self.checkDapi.setChecked(True)
        elif channel_num == 4 and hasattr(self, 'checkCH4'):
            self.checkCH4.setChecked(True)

    def _selected_channel_number(self):
        if self.checkCell.isChecked():
            return 1
        if hasattr(self, 'checkCH2') and self.checkCH2.isChecked():
            return 2
        if self.checkDapi.isChecked():
            return 3
        if hasattr(self, 'checkCH4') and self.checkCH4.isChecked():
            return 4
        self.checkCell.setChecked(True)
        return 1

    def _set_current_display_image(self):
        if self.channel_images:
            self.img_org_2ch = self.channel_images.get(self.current_display_channel, next(iter(self.channel_images.values())))
            img_tmp = c_util.convert_uint8(self.img_org_2ch)
            self.img_org_rgb = img_tmp
            self.img_org_2ch_rgb = [c_util.convert_gray2bgr(img_tmp[:, :, i]) for i in range(3)]

    #Define los colores de las detecciones
    def _render_overlay(self, base_img):
        img_pil = Image.fromarray(base_img.copy())
        draw = ImageDraw.Draw(img_pil)

        for x, y in self.coords_nucleos:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="red", width=1)

        for x, y in self.coords_parasitos:
            draw.line((x - 3, y, x + 3, y), fill="yellow", width=1)
            draw.line((x, y - 3, x, y + 3), fill="yellow", width=1)

        for x1, y1, x2, y2 in self.association_lines:
            draw.line((x1, y1, x2, y2), fill="cyan", width=1)

        return np.array(img_pil)

    def _build_association_lines(self, asociaciones, coords_ch1):
        lines = []
        for idx_celula, pars in asociaciones.items():
            if idx_celula >= len(coords_ch1):
                continue
            x2, y2 = coords_ch1[idx_celula]
            for x1, y1 in pars:
                lines.append((float(x1), float(y1), float(x2), float(y2)))
        return lines

    def result_count(self, data_result):
        self.res_par.setText(str(round(data_result[0], 1)))
        self.res_macrop.setText(str(round(data_result[1], 1)))
        self.res_inf.setText(str(round(data_result[2], 1)))
        if hasattr(self, 'res_sanas'):
            self.res_sanas.setText("")
        if hasattr(self, 'res_media'):
            self.res_media.setText("")
        if hasattr(self, 'res_pct'):
            self.res_pct.setText("")


    def count_img(self):
        self.analizar_imagen_actual()

    def update_view(self, *args):
        if self.img_org_2ch is None:
            return

        self.current_display_channel = self._selected_channel_number()
        self._set_current_display_image()
        channel = max(0, min(2, self.current_display_channel - 1))

        if self.checkView_maps.isChecked() and self.density_maps is not None:
            base_img = c_util.merge_img_map(self.img_org_2ch_rgb[channel], self.density_maps)
        else:
            base_img = self.img_org_rgb

        if hasattr(self, 'checkBW') and self.checkBW.isChecked():
            base_img = np.array(Image.fromarray(base_img).convert("L").convert("RGB"))

        self.img_view = self._render_overlay(base_img)

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
        # Detectar si estamos en modo multicanal (r*_CH*)
        if self.r_number is not None and (self.img_ch1 is not None or self.img_ch3 is not None):
            self._analizar_multicanal()
        elif self.img_org_2ch is not None:
            # Modo tradicional
            self._analizar_tradicional()
        else:
            self.state_process.setText("Primero carga una imagen.")
    
    def _analizar_multicanal(self):
        """Analiza imágenes con patrón r*_CH* procesando CH1 y CH3 pero mostrando en la imagen cargada"""
        self.state_process.setText(f"Analizando {self.r_number}...")
        
        n_nucleos = 0
        n_parasitos = 0
        pred_ch1 = None
        pred_ch3 = None
        coords_ch1 = np.empty((0, 2), dtype=float)
        coords_ch3 = np.empty((0, 2), dtype=float)
        
        # Procesar CH1 (núcleos) si está disponible
        if self.img_ch1 is not None:
            maps_ch1, _ = ficron.count_cells(img_org=self.img_ch1, channel=0, return_map=True)
            pred_ch1 = maps_ch1[1]  # mask_c (núcleos)
            n_nucleos, centros_ch1_yx = c_util.detectar_parasitos_maxima_parametrizable(
                pred_ch1, sigma_suavizado=5, min_distance=21, percentil_fondo=20
            )
            coords_ch1 = np.array([(float(x), float(y)) for (y, x) in centros_ch1_yx], dtype=float)
        else:
            self.state_process.setText(f"Advertencia: No se encontró CH1 para {self.r_number}")
        
        # Procesar CH3 (parásitos) si está disponible
        if self.img_ch3 is not None:
            maps_ch3, _ = ficron.count_cells(img_org=self.img_ch3, channel=2, return_map=True)
            pred_ch3 = maps_ch3[0]  # mask_p (parásitos)
            
            # Preparar para detección de parásitos
            img_ch3_tmp = c_util.convert_uint8(self.img_ch3)
            img_ch3_rgb = [c_util.convert_gray2bgr(img_ch3_tmp[:, :, i]) for i in range(3)]
            
            n_parasitos, centros_ch3_xy = c_util.contar_parasitos_dbscan(img_ch3_rgb[2], pred_ch3)
            coords_ch3 = np.array(centros_ch3_xy, dtype=float) if len(centros_ch3_xy) else np.empty((0, 2), dtype=float)
        else:
            self.state_process.setText(f"Advertencia: No se encontró CH3 para {self.r_number}")
        
        # Calcular células infectadas y visualizar en la imagen cargada (img_org_rgb)
        infectadas = 0
        sanas = n_nucleos
        media_parasitos = 0.0
        pct_infectadas = 0.0
        asociaciones = {}
        
        # Asociación parásito -> núcleo si tenemos ambos
        if len(coords_ch1) > 0 and len(coords_ch3) > 0:
            draw_aux = ImageDraw.Draw(Image.fromarray(self.img_org_rgb.copy()))
            asociaciones = c_util.asociar_por_clustering(coords_ch1, coords_ch3, draw_aux)
            infectadas, sanas, media_parasitos, pct_infectadas = c_util.resumen(asociaciones, len(coords_ch1))

        self.coords_nucleos = coords_ch1
        self.coords_parasitos = coords_ch3
        self.association_lines = self._build_association_lines(asociaciones, coords_ch1)
        self.update_view()
        
        # Actualizar resultados
        self.res_par.setText(str(n_parasitos))
        self.res_macrop.setText(str(n_nucleos))
        self.res_inf.setText(str(infectadas))
        if hasattr(self, 'res_sanas'):
            self.res_sanas.setText(str(sanas))
        if hasattr(self, 'res_media'):
            self.res_media.setText(f"{media_parasitos:.2f}")
        if hasattr(self, 'res_pct'):
            self.res_pct.setText(f"{pct_infectadas:.1f}%")
        
        self.state_process.setText(
            f"{self.r_number} | Núcleos (CH1): {n_nucleos}, Parásitos (CH3): {n_parasitos}, "
            f"Infectadas: {infectadas}, Sanas: {sanas}, Media: {media_parasitos:.2f}"
        )
    
    def _analizar_tradicional(self):
        """Análisis tradicional con imagen multicanal"""
        self.state_process.setText("Analizando...")

        # 1) Predicciones (núcleos/ parásitos)
        pred_ch1, pred_ch4 = self._infer_maps()

        # 2) Detección de núcleos (sale en formato (y, x))
        n_nucleos, centros_ch1_yx = c_util.detectar_parasitos_maxima_parametrizable(pred_ch1, sigma_suavizado=5, min_distance=21, percentil_fondo=20)
        coords_ch1 = np.array([(float(x), float(y)) for (y, x) in centros_ch1_yx], dtype=float)

        # 3) Detección de parásitos (sale en formato (x, y))
        n_parasitos, centros_ch4_xy = c_util.contar_parasitos_dbscan( self.img_org_2ch_rgb[2], pred_ch4 )
        coords_ch4 = np.array(centros_ch4_xy, dtype=float) if len(centros_ch4_xy) else np.empty((0, 2), dtype=float)

        # 4) Asignación parásito -> núcleo (líneas cian)
        if len(coords_ch1) > 0 and len(coords_ch4) > 0:
            draw_aux = ImageDraw.Draw(Image.fromarray(self.img_org_rgb.copy()))
            asociaciones = c_util.asociar_por_clustering(coords_ch1, coords_ch4, draw_aux)
            infectadas, sanas, media_parasitos, pct_infectadas = c_util.resumen(asociaciones, len(coords_ch1))
        else:
            asociaciones = {}
            infectadas, sanas, media_parasitos = 0, len(coords_ch1), 0.0
            pct_infectadas = 0.0

        self.coords_nucleos = coords_ch1
        self.coords_parasitos = coords_ch4
        self.association_lines = self._build_association_lines(asociaciones, coords_ch1)
        self.update_view()

        # Actualizar resultados
        self.res_par.setText(str(n_parasitos))
        self.res_macrop.setText(str(n_nucleos))
        self.res_inf.setText(str(infectadas))
        if hasattr(self, 'res_sanas'):
            self.res_sanas.setText(str(sanas))
        if hasattr(self, 'res_media'):
            self.res_media.setText(f"{media_parasitos:.2f}")
        if hasattr(self, 'res_pct'):
            self.res_pct.setText(f"{pct_infectadas:.1f}%")

        self.state_process.setText(
            f"Listo | Núcleos: {n_nucleos}, Parásitos: {n_parasitos}, "
            f"Infectadas: {infectadas}, Sanas: {sanas}, Media: {media_parasitos:.2f}"
        )

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()