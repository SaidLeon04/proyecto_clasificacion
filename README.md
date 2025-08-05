# Control de Mouse por Gestos de Mano

Sistema de control de mouse basado en visión por computadora que permite la navegación del cursor y clics sin contacto a través del reconocimiento de gestos de mano en tiempo real utilizando entrada de cámara web.

## Descripción General

Este proyecto implementa una interfaz de control de mouse sin contacto utilizando técnicas de visión por computadora y aprendizaje automático. El sistema captura gestos de mano a través de una cámara web, los procesa utilizando la detección de puntos clave de mano de MediaPipe, y traduce gestos específicos en acciones del mouse a través de un modelo de clasificación entrenado.

## Características Principales

- Reconocimiento de gestos de mano en tiempo real
- Control de movimiento del cursor mediante gestos de mano abierta
- Funcionalidad de clic izquierdo mediante gestos de mano cerrada
- Clasificación de gestos basada en aprendizaje automático
- Compatibilidad multiplataforma
- Sistema de respuesta de baja latencia

## Stack Tecnológico

**Visión por Computadora**: OpenCV para captura de video y procesamiento de imágenes  
**Detección de Manos**: MediaPipe para extracción de puntos clave de la mano  
**Aprendizaje Automático**: Scikit-learn con Regresión Logística para clasificación de gestos  
**Control del Sistema**: PyAutoGUI para automatización del mouse  
**Procesamiento de Datos**: NumPy y Pandas para ingeniería de características

## Instalación

Clonar el repositorio:
```bash
git clone https://github.com/SaidLeon04/proyecto_clasificacion.git
cd proyecto_clasificacion
```

Instalar las dependencias requeridas:
```bash
pip install opencv-python==4.7.0.72 mediapipe==0.10.21 pyautogui==0.9.54 numpy==1.26.4 scikit-learn==1.7.1 joblib==1.5.1
```

Alternativamente, instalar desde requirements.txt si está disponible:
```bash
pip install -r requirements.txt
```

## Uso

### Inicio Rápido
Ejecutar la aplicación principal desde la carpeta implementacion:
```bash
cd implementacion
python app.py
```

O ejecutar el script principal alternativo:
```bash
python app.py
```

### Entrenamiento de Modelo Personalizado (Opcional)
Los scripts de entrenamiento se encuentran en la carpeta `entrenamiento/` y los de extracción de puntos en `obtener_puntos/`.

Para generar un nuevo dataset y entrenar el modelo, ejecutar los scripts correspondientes desde sus respectivas carpetas.

## Arquitectura del Sistema

### Pipeline de Datos
1. **Captura de Video**: Adquisición de frames en tiempo real desde la cámara web
2. **Detección de Manos**: MediaPipe extrae 21 puntos clave de la mano por frame
3. **Extracción de Características**: Conversión de coordenadas de puntos clave a vectores de características normalizados
4. **Clasificación**: Modelo de regresión logística predice el tipo de gesto (abierta/cerrada)
5. **Ejecución de Acciones**: PyAutoGUI traduce las predicciones a eventos del mouse del sistema

### Reconocimiento de Gestos
- **Mano Abierta (Clase 1)**: Activa el movimiento del cursor basado en la posición de la palma
- **Mano Cerrada (Clase 0)**: Ejecuta acción de clic izquierdo del mouse

### Rendimiento del Modelo
El modelo de clasificación logra alta precisión a través de la ingeniería de características de coordenadas de puntos clave de la mano, proporcionando reconocimiento confiable de gestos en diferentes condiciones de iluminación y orientaciones de la mano.

## Estructura del Proyecto

```
proyecto_clasificacion/
├── entrenamiento/             # Scripts y archivos de entrenamiento del modelo
├── obtener_puntos/            # Scripts para extracción de puntos clave
│   ├── closeHands/           # Imágenes de gestos de mano cerrada
│   └── openHands/            # Imágenes de gestos de mano abierta
├── implementacion/           # Aplicación principal
│   ├── app.py               # Punto de entrada de la aplicación
│   └── modelo_manos.joblib  # Modelo entrenado serializado
├── app.py                   # Script principal alternativo
├── hands.csv               # Dataset de características de entrenamiento
├── hands.py                # Utilidades de procesamiento de manos
└── README.md               # Documentación del proyecto
```

## Configuración

### Parámetros del Sistema
La aplicación puede personalizarse modificando parámetros en `implementacion/app.py` o `app.py`:

```python
# Configuración de cámara
cap = cv2.VideoCapture(0)  # Cambiar índice para diferentes cámaras

# Configuración MediaPipe
hands = mp_hands.Hands(max_num_hands=1)  # Detección de una sola mano

# Mapeo de pantalla
screen_w, screen_h = pyautogui.size()  # Detección automática de resolución de pantalla
```

### Re-entrenamiento del Modelo
Para mejorar la precisión con datos personalizados:

1. Agregar imágenes de gestos a los directorios `obtener_puntos/openHands/` y `obtener_puntos/closeHands/`
2. Ejecutar scripts en la carpeta `entrenamiento/` para generar nuevo dataset
3. Entrenar modelo actualizado
4. Reemplazar archivo `implementacion/modelo_manos.joblib`

## Dependencias Principales

Las siguientes librerías son esenciales para el funcionamiento del sistema:

- **opencv-python (4.7.0.72)**: Procesamiento de video e imágenes
- **mediapipe (0.10.21)**: Detección y seguimiento de manos
- **pyautogui (0.9.54)**: Control automático del mouse y teclado
- **numpy (1.26.4)**: Computación numérica y manejo de arrays
- **scikit-learn (1.7.1)**: Algoritmos de aprendizaje automático
- **joblib (1.5.1)**: Serialización eficiente de modelos

## Requisitos Técnicos

**Hardware**:
- Cámara web o cámara integrada
- Mínimo 4GB RAM
- CPU con soporte SSE4.2

**Software**:
- Python 3.7 o superior
- Sistema operativo: Windows, macOS o Linux

## Consideraciones de Rendimiento

El sistema está optimizado para rendimiento en tiempo real con tasas de procesamiento de frames adecuadas para control responsivo del mouse. El uso de memoria se minimiza através de extracción eficiente de características y cache de predicciones del modelo.

## Solución de Problemas

**Problemas de Acceso a Cámara**:
Asegurar que ninguna otra aplicación esté usando la cámara. Probar diferentes índices de cámara (0, 1, 2) si el predeterminado falla.

**Errores de Importación**:
Verificar que todas las dependencias estén instaladas correctamente. Usar entorno virtual para gestión limpia de paquetes.

**Errores de Carga del Modelo**:
Asegurar que `modelo_manos.joblib` existe en el directorio del proyecto. Re-ejecutar script de entrenamiento si el archivo falta.

## Contribuciones

Las contribuciones son bienvenidas. Por favor seguir el flujo de trabajo estándar de Git:

1. Fork del repositorio
2. Crear rama de funcionalidad
3. Implementar cambios con pruebas apropiadas
4. Enviar pull request con descripción detallada

---

Para preguntas, reportes de errores o solicitudes de funcionalidades, por favor abrir un issue en el repositorio.
