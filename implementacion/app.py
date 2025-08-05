import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from joblib import load

# Cargar modelo entrenado
model = load("modelo_manos.joblib")

# Inicializar MediaPipe y configuración
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Estado de clic
click_down = False

# Iniciar cámara
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontal y convertir color
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar la mano
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer coordenadas x, y normalizadas
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y])

            # Convertir a DataFrame
            features_np = np.array(features).reshape(1, -1)

            # Predecir clase: 0 = cerrada, 1 = abierta
            prediction = model.predict(features_np)[0]

            if prediction == 1:
                # Mano abierta: mover cursor
                palm = hand_landmarks.landmark[9]
                x = int(palm.x * screen_w)
                y = int(palm.y * screen_h)
                pyautogui.moveTo(x, y)
                click_down = False  # Resetear estado de clic
                cv2.putText(frame, "Mano Abierta", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Mano cerrada: hacer clic izquierdo una vez
                if not click_down:
                    click_down = True
                    pyautogui.click()
                    cv2.putText(frame, "Click Izquierdo", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Control de Mano", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
