import os
import cv2
import csv
import mediapipe as mp

# Definir nombres de los landmarks
nombres_puntos = [
    "muñeca",
    "base_pulgar", "intermedia_pulgar", "final_pulgar", "punta_pulgar",
    "base_indice", "intermedia_indice", "final_indice", "punta_indice",
    "base_medio", "intermedia_medio", "final_medio", "punta_medio",
    "base_anular", "intermedia_anular", "final_anular", "punta_anular",
    "base_meñique", "intermedia_meñique", "final_meñique", "punta_meñique"
]

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Carpeta con imágenes
# carpeta = "openHands"
carpeta = "closeHands" # cambiar nombre de la carpeta
imagenes = [f for f in os.listdir(carpeta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Archivo de salida CSV
csv_salida = "datos_manos_cerradas.csv" # cambiar nombre del csv (opcional)

with open(csv_salida, mode='w', newline='') as archivo_csv:
    escritor = csv.writer(archivo_csv)

    # Encabezado dinámico: punto_x, punto_y, ..., clase
    encabezado = []
    for nombre in nombres_puntos:
        encabezado.append(f"{nombre}_x")
        encabezado.append(f"{nombre}_y")
    encabezado.append("clase")
    escritor.writerow(encabezado)

    for imagen_nombre in imagenes:
        ruta = os.path.join(carpeta, imagen_nombre)
        imagen = cv2.imread(ruta)

        if imagen is None:
            print(f"Error al leer {ruta}")
            continue

        # Convertir a RGB
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        if resultado.multi_hand_landmarks:
            for hand_landmarks in resultado.multi_hand_landmarks:
                fila = []
                for punto in hand_landmarks.landmark:
                    fila.extend([punto.x, punto.y])
                if len(fila) == 42:
                    fila.append("cerrada")  # cambiar a 0 si es cerrada | cambiar a 1 si es abierta
                    escritor.writerow(fila)
                else:
                    print(f"Puntos incompletos en: {imagen_nombre}")
        else:
            print(f"No se detectó mano en: {imagen_nombre}")

print("CSV generado con éxito: datos_manos_abiertas.csv")
