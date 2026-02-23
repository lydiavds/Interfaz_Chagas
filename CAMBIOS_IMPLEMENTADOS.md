# Cambios Implementados - Detección Automática de Imágenes r*

## Resumen
Se ha implementado una funcionalidad que permite detectar automáticamente imágenes con formato `r<número>_CH<X>.tif` y cargar automáticamente los canales **CH1** (para detección de núcleos) y **CH3** (para detección de parásitos), mostrando ambas detecciones independientemente del canal que se esté visualizando.

## Cambios Realizados en `main.py`

### 1. Nuevas funciones auxiliares (líneas 26-57)

#### `extract_replica_number(filename)`
- Extrae el número de réplica del nombre de archivo usando regex
- Patrón: `r(\d+)_CH\d`
- Ej: `r123_CH1.tif` → `123`
- Retorna `None` si no es una imagen `r*`

#### `build_channel_filepath(filepath, target_channel)`
- Construye la ruta para cargar un canal específico
- Patrón: Transforma `r123_CH1.tif` con `target_channel=3` → `r123_CH3.tif`
- Mantiene la extensión del archivo
- Retorna `None` si el patrón no coincide

### 2. Nuevas propiedades en `MainWindow.__init__()` (líneas 69-74)

```python
self.current_filepath = None      # Rastrea la ruta del archivo actual
self.is_replica_image = False     # Indica si es imagen r<numero>_CH<X>
self.ch1_image = None            # Almacena la imagen del canal CH1 (núcleos)
self.ch3_image = None            # Almacena la imagen del canal CH3 (parásitos)
```

### 3. Modificación de `open_imag()` (líneas 84-141)

**Comportamiento nuevo:**
1. Detecta si el archivo tiene el patrón `r<número>_CH<X>`
2. Si es una imagen `r*`:
   - Automáticamente carga el archivo `r<número>_CH1.tif` (núcleos)
   - Automáticamente carga el archivo `r<número>_CH3.tif` (parásitos)
   - Muestra un mensaje indicando que se detectaron ambos canales
3. Si no es una imagen `r*`, funciona como antes (carga el archivo seleccionado)

**Manejo de errores:**
- Si falla la carga de CH1 o CH3, muestra un mensaje de error específico

### 4. Modificación de `_infer_maps()` (líneas 193-217)

**Nuevo flujo para imágenes `r*`:**
1. Si es imagen `r*` y se cargaron CH1 y CH3:
   - Procesa CH1 con el modelo para obtener el mapa de núcleos
   - Procesa CH3 con el modelo para obtener el mapa de parásitos
2. Si no es imagen `r*`, usa el comportamiento original (imagen cargada)

**Ventaja:** Cada canal se analiza con su modelo específico, mejorando la precisión

### 5. Modificación de `analizar_imagen_actual()` (líneas 219-304)

**Cambios principales:**

- **Detección de parásitos:** Para imágenes `r*`, usa específicamente `ch3_image` para la detección de parásitos
- **Visualización:** Siempre muestra ambas detecciones (núcleos en rojo, parásitos en amarillo)
- **Mensajes de estado:** Incluye información sobre el tipo de análisis:
  - Imagen `r*`: `"Listo (imagen r*) | Núcleos (CH1): X, Parásitos (CH3): Y, ..."`
  - Imagen normal: `"Listo | Núcleos: X, Parásitos: Y, ..."`

## Flujo de Uso

### Cuando carga una imagen `r123_CH1.tif`:
1. El sistema detecta automáticamente que es `r123` 
2. Busca y carga `r123_CH1.tif` (si no existe, muestra error)
3. Busca y carga `r123_CH3.tif` (si no existe, muestra error)
4. Muestra el mensaje: `"Detectada imagen r* - Canales CH1 y CH3 cargados automáticamente"`
5. Al analizar:
   - Usa CH1 para detectar núcleos
   - Usa CH3 para detectar parásitos
   - Muestra ambas detecciones en el mismo overlay

### Cuando carga una imagen normal (no `r*`):
1. El sistema funciona igual que antes
2. No intenta cargar canales adicionales
3. Analiza la imagen cargada como lo hacía anteriormente

## Requisitos

- Los canales **CH1.tif** y **CH3.tif** deben existir en la misma carpeta
- Deben tener el formato de nombre exacto: `r<número>_CH1.tif` y `r<número>_CH3.tif`
- Los modelos en `ficron.py` deben estar disponibles

## Ventajas del Nuevo Sistema

✅ Automatización: No requiere seleccionar manualmente cada canal  
✅ Consistencia: Siempre analiza los mismos canales (CH1 para núcleos, CH3 para parásitos)  
✅ Precisión mejorada: Cada canal se procesa con su modelo específico  
✅ Flexibilidad: Permite visualizar cualquier canal mientras mantiene las detecciones  
✅ Retrocompatibilidad: Las imágenes sin patrón `r*` funcionan igual que antes

## Pruebas Recomendadas

1. Cargar una imagen `r123_CH1.tif` → Debe cargar automáticamente CH1 y CH3
2. Cargar una imagen `r123_CH2.tif` → Debe cargar automáticamente CH1 y CH3
3. Cargar una imagen `r123_CH3.tif` → Debe cargar automáticamente CH1 y CH3
4. Cargar una imagen `r123_CH4.tif` → Debe cargar automáticamente CH1 y CH3
5. Analizar la imagen → Debe mostrar ambas detecciones con mensaje "(imagen r*)"
6. Cargar una imagen que NO sea `r*` → Debe funcionar como antes
