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
import seaborn as sns


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
        self.zoom_factor = 1.0
        self.zoom_step = 0.25
        self.zoom_min = 1.0
        self.zoom_max = 20.0
        
        # Nuevas variables para manejo de canales separados
        self.img_ch1 = None  # Imagen del canal CH1 (núcleos)
        self.img_ch3 = None  # Imagen del canal CH3 (parásitos)
        self.r_number = None  # Número r compartido entre ambas imágenes
        self.current_filename = None
        self.channel_images = {}  # {1: np.array, 2: np.array, 3: np.array, 4: np.array}
        self.channel_density_maps = {}  # {1: [mask_p, mask_c, mask_i], 3: [...]}
        self.current_display_channel = 1

        self.coords_nucleos = np.empty((0, 2), dtype=float)
        self.coords_parasitos = np.empty((0, 2), dtype=float)
        self.association_lines = []  # [(x1, y1, x2, y2), ...]

        if hasattr(self, 'checkShowCount'):
            self.checkShowCount.setEnabled(False)
            self.checkShowCount.setChecked(False)

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

        if hasattr(self, 'scrollAreaImage'):
            viewport = self.scrollAreaImage.viewport()
            viewport.installEventFilter(self)
            viewport.grabGesture(QtCore.Qt.PinchGesture)
            viewport.setAttribute(QtCore.Qt.WA_AcceptTouchEvents, True)

        self._update_count_button_state()
        self._update_zoom_controls_state(False)
        self._update_zoom_percent_label()

    def _update_count_button_state(self):
        has_image = self.img_org_2ch is not None
        if hasattr(self, 'Button_try'):
            self.Button_try.setEnabled(has_image)
        self._update_image_controls_state(has_image)

    def _update_image_controls_state(self, has_image):
        if hasattr(self, 'checkCell'):
            self.checkCell.setEnabled(has_image)
        if hasattr(self, 'checkCH2'):
            self.checkCH2.setEnabled(has_image)
        if hasattr(self, 'checkDapi'):
            self.checkDapi.setEnabled(has_image)
        if hasattr(self, 'checkCH4'):
            self.checkCH4.setEnabled(has_image)
        if hasattr(self, 'checkBW'):
            self.checkBW.setEnabled(has_image)

        self._update_zoom_controls_state(has_image)

        if not has_image:
            self.zoom_factor = 1.0
            self._update_zoom_percent_label()
            if hasattr(self, 'checkShowCount'):
                self.checkShowCount.setEnabled(False)
                self.checkShowCount.setChecked(False)
            if hasattr(self, 'checkView_maps'):
                self.checkView_maps.setEnabled(False)
                self.checkView_maps.setChecked(False)

    def _update_zoom_controls_state(self, has_image):
        if hasattr(self, 'btn_zoom_in'):
            self.btn_zoom_in.setEnabled(has_image and self.zoom_factor < self.zoom_max)
        if hasattr(self, 'btn_zoom_out'):
            self.btn_zoom_out.setEnabled(has_image and self.zoom_factor > self.zoom_min)

    def _update_zoom_percent_label(self):
        if hasattr(self, 'zoom_percent'):
            self.zoom_percent.setText(f"{int(round(self.zoom_factor * 100))}%")

    def _clear_count_results(self):
        if hasattr(self, 'res_par'):
            self.res_par.setText("")
        if hasattr(self, 'res_macrop'):
            self.res_macrop.setText("")
        if hasattr(self, 'res_inf'):
            self.res_inf.setText("")
        if hasattr(self, 'res_sanas'):
            self.res_sanas.setText("")
        if hasattr(self, 'res_media'):
            self.res_media.setText("")
        if hasattr(self, 'res_pct'):
            self.res_pct.setText("")


    def set_view_image(self, img_npRGB):
        source_pixmap = img_np2qt(img_npRGB)

        if hasattr(self, 'scrollAreaImage'):
            viewport_size = self.scrollAreaImage.maximumViewportSize()
            view_w = max(1, viewport_size.width())
            view_h = max(1, viewport_size.height())
        else:
            view_w = max(1, self.view_image.maximumWidth())
            view_h = max(1, self.view_image.maximumHeight())

        fit_scale = min(view_w / source_pixmap.width(), view_h / source_pixmap.height())
        display_scale = max(0.01, fit_scale * self.zoom_factor)
        zoom_w = max(1, int(round(source_pixmap.width() * display_scale)))
        zoom_h = max(1, int(round(source_pixmap.height() * display_scale)))
        transformation = (QtCore.Qt.SmoothTransformation
                  if display_scale < 1.0
                  else QtCore.Qt.FastTransformation)
        img_view = source_pixmap.scaled(zoom_w, zoom_h, QtCore.Qt.KeepAspectRatio, transformation)
        self.view_image.resize(img_view.size())
        self.view_image.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.view_image.setPixmap(img_view)

    #Para mantener el centro de la vista al hacer zoom
    def _get_view_center_ratios(self):
        if not hasattr(self, 'scrollAreaImage'):
            return None
        hbar = self.scrollAreaImage.horizontalScrollBar()
        vbar = self.scrollAreaImage.verticalScrollBar()
        viewport = self.scrollAreaImage.viewport()

        img_w = max(1, self.view_image.width())
        img_h = max(1, self.view_image.height())
        center_x = hbar.value() + (viewport.width() / 2.0)
        center_y = vbar.value() + (viewport.height() / 2.0)

        ratio_x = max(0.0, min(1.0, center_x / img_w))
        ratio_y = max(0.0, min(1.0, center_y / img_h))
        return ratio_x, ratio_y

    def _restore_view_center_ratios(self, center_ratios):
        if center_ratios is None or not hasattr(self, 'scrollAreaImage'):
            return

        ratio_x, ratio_y = center_ratios
        hbar = self.scrollAreaImage.horizontalScrollBar()
        vbar = self.scrollAreaImage.verticalScrollBar()
        viewport = self.scrollAreaImage.viewport()

        target_x = int(round((ratio_x * self.view_image.width()) - (viewport.width() / 2.0)))
        target_y = int(round((ratio_y * self.view_image.height()) - (viewport.height() / 2.0)))

        target_x = max(hbar.minimum(), min(target_x, hbar.maximum()))
        target_y = max(vbar.minimum(), min(target_y, vbar.maximum()))
        hbar.setValue(target_x)
        vbar.setValue(target_y)

    def _get_view_anchor_ratios(self, anchor_pos=None):
        if not hasattr(self, 'scrollAreaImage'):
            return None
        hbar = self.scrollAreaImage.horizontalScrollBar()
        vbar = self.scrollAreaImage.verticalScrollBar()
        viewport = self.scrollAreaImage.viewport()

        if anchor_pos is None:
            anchor_x = viewport.width() / 2.0
            anchor_y = viewport.height() / 2.0
        else:
            anchor_x = float(anchor_pos.x())
            anchor_y = float(anchor_pos.y())

        img_w = max(1, self.view_image.width())
        img_h = max(1, self.view_image.height())
        content_x = hbar.value() + anchor_x
        content_y = vbar.value() + anchor_y

        ratio_x = max(0.0, min(1.0, content_x / img_w))
        ratio_y = max(0.0, min(1.0, content_y / img_h))
        return ratio_x, ratio_y

    def _restore_view_anchor_ratios(self, anchor_ratios, anchor_pos=None):
        if anchor_ratios is None or not hasattr(self, 'scrollAreaImage'):
            return

        ratio_x, ratio_y = anchor_ratios
        hbar = self.scrollAreaImage.horizontalScrollBar()
        vbar = self.scrollAreaImage.verticalScrollBar()
        viewport = self.scrollAreaImage.viewport()

        if anchor_pos is None:
            anchor_x = viewport.width() / 2.0
            anchor_y = viewport.height() / 2.0
        else:
            anchor_x = float(anchor_pos.x())
            anchor_y = float(anchor_pos.y())

        target_x = int(round((ratio_x * self.view_image.width()) - anchor_x))
        target_y = int(round((ratio_y * self.view_image.height()) - anchor_y))

        target_x = max(hbar.minimum(), min(target_x, hbar.maximum()))
        target_y = max(vbar.minimum(), min(target_y, vbar.maximum()))
        hbar.setValue(target_x)
        vbar.setValue(target_y)

    def _set_zoom_factor(self, new_zoom_factor, anchor_pos=None):
        if self.img_org_2ch is None or self.img_view is None:
            return

        clamped_zoom = max(self.zoom_min, min(float(new_zoom_factor), self.zoom_max))
        if abs(clamped_zoom - self.zoom_factor) < 1e-9:
            self._update_zoom_controls_state(True)
            return

        anchor_ratios = self._get_view_anchor_ratios(anchor_pos)
        self.zoom_factor = clamped_zoom
        self._update_zoom_percent_label()
        self._update_zoom_controls_state(True)
        self.set_view_image(self.img_view)
        self._restore_view_anchor_ratios(anchor_ratios, anchor_pos)

    def _is_mouse_wheel_event(self, wheel_event):
        if wheel_event.source() != QtCore.Qt.MouseEventNotSynthesized:
            return False

        if wheel_event.phase() != QtCore.Qt.NoScrollPhase:
            return False

        if not wheel_event.pixelDelta().isNull():
            return False

        return abs(wheel_event.angleDelta().y()) >= 120

    def eventFilter(self, obj, event):
        if hasattr(self, 'scrollAreaImage') and obj is self.scrollAreaImage.viewport() and self.img_org_2ch is not None:
            if event.type() == QtCore.QEvent.Wheel and self._is_mouse_wheel_event(event):
                delta_y = event.angleDelta().y()
                if delta_y == 0:
                    delta_y = event.pixelDelta().y()
                if delta_y != 0:
                    direction = 1 if delta_y > 0 else -1
                    target_zoom = self.zoom_factor + (direction * self.zoom_step)
                    self._set_zoom_factor(target_zoom, event.pos())
                    return True

            if event.type() == QtCore.QEvent.Gesture:
                pinch = event.gesture(QtCore.Qt.PinchGesture)
                if pinch is not None and (pinch.changeFlags() & QtWidgets.QPinchGesture.ScaleFactorChanged):
                    scale_factor = pinch.scaleFactor()
                    if scale_factor > 0:
                        center_pos = pinch.centerPoint().toPoint()
                        target_zoom = self.zoom_factor * float(scale_factor)
                        self._set_zoom_factor(target_zoom, center_pos)
                        return True

        return super().eventFilter(obj, event)

    def zoom_in(self):
        if self.img_org_2ch is None:
            return
        self._set_zoom_factor(self.zoom_factor + self.zoom_step)

    def zoom_out(self):
        if self.img_org_2ch is None:
            return
        self._set_zoom_factor(self.zoom_factor - self.zoom_step)

    def open_imag(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "Lote_final_procesado/", "Images (*.tif *.lif *.png *.jpg *.jpeg)",
                                                  options=options)
        prev_r_number = self.r_number
        
        #Añadido para que cuando no se seleccione ninguna imagen, no se tome la imagen seleccionada como "", y simplemente cancele la seleccion
        if not fileName:
            self.state_process.setText('Selección cancelada.')  
            self._update_count_button_state()
            return

        self.path_dir = os.path.dirname(fileName)
        self.text_path_dir.setText(fileName)
        self.zoom_factor = 1.0
        self._update_zoom_percent_label()
        
        # Extraer nombre del archivo sin extensión
        basename = os.path.basename(fileName)
        name_without_ext = os.path.splitext(basename)[0]
        dir_path = os.path.dirname(fileName)
        
        # Detectar cualquier nombre terminado en _CH1, _CH2, _CH3 o _CH4.
        pattern = re.match(r'(?P<base>.+)_CH(?P<channel>[1-4])$', name_without_ext, re.IGNORECASE)
        same_r = False
        
        if pattern:
            # Imagen con patrón detectado
            new_r_number = pattern.group('base')
            same_r = (prev_r_number is not None and prev_r_number == new_r_number)
            if prev_r_number is not None and prev_r_number != new_r_number:
                self._clear_count_results()
            self.r_number = new_r_number
            current_channel = int(pattern.group('channel'))
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
            if not same_r:
                self.channel_density_maps = {}
            supported_extensions = {'.tif', '.tiff', '.lif', '.png', '.jpg', '.jpeg'}
            for channel_num in (1, 2, 3, 4):
                channel_stem = f"{self.r_number}_CH{channel_num}".casefold()
                channel_path = next(
                    (
                        os.path.join(dir_path, filename)
                        for filename in os.listdir(dir_path)
                        if os.path.splitext(filename)[0].casefold() == channel_stem
                        and os.path.splitext(filename)[1].casefold() in supported_extensions
                    ),
                    None
                )
                if channel_path is not None:
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
            # Imagen sin patrón, cargar de forma tradicional (es decir, se hace toda la detección en la misma imagen)
            if prev_r_number is not None:
                self._clear_count_results()
            self.r_number = None
            self.img_ch1 = None
            self.img_ch3 = None
            self.channel_images = {}
            self.channel_density_maps = {}
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

        if not same_r:
            self.coords_nucleos = np.empty((0, 2), dtype=float)
            self.coords_parasitos = np.empty((0, 2), dtype=float)
            self.association_lines = []

            if hasattr(self, 'checkShowCount'):
                self.checkShowCount.setEnabled(False)
                self.checkShowCount.setChecked(False)
            
            self.checkView_maps.setChecked(False)
            self.density_maps = None
        self._update_count_button_state()
        self._update_zoom_controls_state(self.img_org_2ch is not None)
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
            self.density_maps = self.channel_density_maps.get(self.current_display_channel)

    def _update_density_map_toggle(self):
        if not hasattr(self, 'checkView_maps'):
            return
        has_map = (self.img_org_2ch is not None) and (self.density_maps is not None)
        self.checkView_maps.setEnabled(has_map)
        if not has_map and self.checkView_maps.isChecked():
            self.checkView_maps.setChecked(False)

    def _render_density_map(self, maps):
        if maps is None or self.img_org_rgb is None:
            return self.img_org_rgb

        if self.current_display_channel == 1:
            density_map = maps[1]  # núcleos
        elif self.current_display_channel == 3:
            density_map = maps[0]  # parásitos
        else:
            density_map = np.maximum(maps[0], maps[1])

        density_norm = c_util.convert_uint8(density_map).astype(np.float32) / 255.0

        
        cmap = sns.color_palette("rocket", as_cmap=True)
        heat_rgb = (cmap(density_norm)[..., :3] * 255).astype(np.uint8)
        return heat_rgb

    #Define los colores de las detecciones
    def _render_overlay(self, base_img):
        img_pil = Image.fromarray(base_img.copy())
        draw = ImageDraw.Draw(img_pil)

        for x, y in self.coords_nucleos:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="white", width=1)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="black", width=4)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="red", width=3)

        for x, y in self.coords_parasitos:
            draw.line((x - 6, y, x + 6, y), fill="black", width=3)
            draw.line((x, y - 6, x, y + 6), fill="black", width=3)
            draw.line((x - 6, y, x + 6, y), fill="yellow", width=2)
            draw.line((x, y - 6, x, y + 6), fill="yellow", width=2)

        for x1, y1, x2, y2 in self.association_lines:
            draw.line((x1, y1, x2, y2), fill="black", width=3)
            draw.line((x1, y1, x2, y2), fill="cyan", width=2)

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
        self._update_density_map_toggle()
        channel = max(0, min(2, self.current_display_channel - 1))

        if self.checkView_maps.isChecked() and self.density_maps is not None:
            base_img = self._render_density_map(self.density_maps)
        else:
            base_img = self.img_org_rgb

        if hasattr(self, 'checkBW') and self.checkBW.isChecked():
            base_img = np.array(Image.fromarray(base_img).convert("L").convert("RGB"))

        show_count = hasattr(self, 'checkShowCount') and self.checkShowCount.isEnabled() and self.checkShowCount.isChecked()
        self.img_view = self._render_overlay(base_img) if show_count else base_img

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

        # Llamar a count_cells de ficron.py para obtener las predicciones
        maps, resul_predict = ficron.count_cells(img_org=self.img_org_2ch, return_map=True)
        
        # maps = [mask_p, mask_c, mask_i]
        pred_ch1 = maps[1]  # mask_c (núcleos)
        pred_ch4 = maps[0]  # mask_p (parásitos)
        
        return pred_ch1, pred_ch4, maps

    def analizar_imagen_actual(self):
        # Detectar si estamos en modo multicanal (r*_CH*)
        if self.r_number is not None and (self.img_ch1 is not None or self.img_ch3 is not None):
            self._analizar_multicanal()
        elif self.img_org_2ch is not None:
            # Modo tradicional (toda la deteccion en la misma imagen)
            self._analizar_tradicional()
        else:
            self.state_process.setText("Primero carga una imagen.")
    
    #Lo que realmente realiza el analisis total (detecciones y calculos)
    def _analizar_multicanal(self):
        """Analiza imágenes con patrón r*_CH* procesando CH1 y CH3 pero mostrando en la imagen cargada"""
        self.state_process.setText(f"Analizando {self.r_number}...")
        
        n_nucleos = 0
        n_parasitos = 0
        pred_ch1 = None
        pred_ch3 = None
        maps_ch1 = None
        maps_ch3 = None
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
        if hasattr(self, 'checkShowCount'):
            self.checkShowCount.setEnabled(True)
            self.checkShowCount.setChecked(True)
        self.channel_density_maps = {}
        if maps_ch1 is not None:
            self.channel_density_maps[1] = maps_ch1
        if maps_ch3 is not None:
            self.channel_density_maps[3] = maps_ch3
        self.density_maps = self.channel_density_maps.get(self.current_display_channel)
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
    
    #Esto es lo de todo en una imagen, no se deberia usar
    def _analizar_tradicional(self):
        """Análisis tradicional con imagen multicanal"""
        self.state_process.setText("Analizando...")

        # 1) Predicciones y mapas de densidad (núcleos/ parásitos)
        pred_ch1, pred_ch4, maps = self._infer_maps()

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
        if hasattr(self, 'checkShowCount'):
            self.checkShowCount.setEnabled(True)
            self.checkShowCount.setChecked(True)
        self.channel_density_maps = {self.current_display_channel: maps}
        self.density_maps = maps
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