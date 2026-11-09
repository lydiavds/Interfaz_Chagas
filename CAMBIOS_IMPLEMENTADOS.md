# Cambios implementados en la interfaz

## Resumen

La interfaz de FiCRoN permite cargar imágenes de fluorescencia, detectar células y parásitos automáticamente y mostrar los resultados sobre la imagen.

## Funcionalidades principales

- **Carga de imágenes:** admite `.tif`, `.lif`, `.png`, `.jpg` y `.jpeg` desde `Archivos > Abrir`.
- **Gestión de canales:** reconoce archivos con formato `*_CH1` a `*_CH4` y carga automáticamente los canales disponibles del mismo grupo.
- **Compatibilidad:** las imágenes sin nombre de canal se procesan en modo tradicional.
- **Modos de detección:** permite elegir entre los modelos del trabajo previo (**Conteo 1**) y los modelos del TFG (**Conteo 2**).
- **Procesamiento:** divide las imágenes grandes en teselas, añade relleno cuando es necesario y reconstruye los mapas de densidad.
- **Visualización:**
  - selección de canales CH1–CH4;
  - mapa de densidad;
  - imagen en blanco y negro;
  - superposición del conteo;
  - zoom con botones, rueda del ratón y gesto táctil.
- **Detecciones sobre la imagen:**
  - células numeradas y coloreadas según su carga parasitaria;
  - parásitos marcados en amarillo;
  - asociaciones célula-parásito marcadas en cian.
- **Resultados:** muestra células totales, infectadas, no infectadas, parásitos, media de parásitos por célula y porcentaje de infección.
- **Carga parasitaria:** clasifica las células en los rangos `1–15`, `16–50` y `>50` parásitos.
- **Gráficos:** genera gráficos por rangos de infección y para el detalle de `1–15` parásitos por célula.
- **Estado del proceso:** informa de la imagen cargada, los canales disponibles, el progreso del análisis y posibles errores.

La interfaz mantiene el comportamiento anterior para imágenes que no siguen la nomenclatura multicanal.
