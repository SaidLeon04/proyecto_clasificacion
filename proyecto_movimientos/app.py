import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Inicializar MediaPipe y PyAutoGUI
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Desactivar el fail-safe de PyAutoGUI que se activa en las esquinas
pyautogui.FAILSAFE = False

# Resolución de pantalla
screen_width, screen_height = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Configurar la ventana para que sea pequeña y siempre visible
window_name = "Control por mano"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Hacer la ventana pequeña (200x150 píxeles)
cv2.resizeWindow(window_name, 200, 150)

# Posicionar la ventana en la esquina superior izquierda
cv2.moveWindow(window_name, 0, 0)

# Hacer la ventana "always on top" (siempre encima)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

click_down = False
right_click_down = False

def finger_extended(lm, tip_id, pip_id):
    return lm[tip_id].y < lm[pip_id].y  # está más arriba en la imagen (más extendido)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # Verificar si hay 3 dedos extendidos (índice, medio, anular)
            is_open_hand = (
                finger_extended(landmarks, 8, 6) and
                finger_extended(landmarks, 12, 10) and
                finger_extended(landmarks, 16, 14)
            )

            # Mover cursor con la palma (landmark 9)
            if is_open_hand:
                palm = landmarks[9]
                x = int(palm.x * w)
                y = int(palm.y * h)

                # Aumentar sensibilidad usando solo el 60% central de la cámara
                # para cubrir toda la pantalla
                palm_x_sensitive = np.interp(palm.x, [0.2, 0.8], [0, 1])
                palm_y_sensitive = np.interp(palm.y, [0.15, 0.85], [0, 1])
                
                # Limitar valores para evitar movimientos extremos
                palm_x_sensitive = np.clip(palm_x_sensitive, 0, 1)
                palm_y_sensitive = np.clip(palm_y_sensitive, 0, 1)

                x_screen = palm_x_sensitive * screen_width
                y_screen = palm_y_sensitive * screen_height

                pyautogui.moveTo(x_screen, y_screen)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Círculo más pequeño
                # Texto más pequeño para la ventana reducida
                cv2.putText(frame, 'Mouse', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Detectar clic izquierdo por proximidad entre índice y pulgar
            x_index = int(landmarks[8].x * w)
            y_index = int(landmarks[8].y * h)
            x_thumb = int(landmarks[4].x * w)
            y_thumb = int(landmarks[4].y * h)

            distance_index_thumb = np.hypot(x_index - x_thumb, y_index - y_thumb)

            cv2.circle(frame, (x_index, y_index), 3, (255, 0, 255), -1)  # Índice
            cv2.circle(frame, (x_thumb, y_thumb), 3, (0, 255, 255), -1)  # Pulgar
            cv2.line(frame, (x_index, y_index), (x_thumb, y_thumb), (255, 255, 0), 1)

            # Detectar clic derecho por proximidad entre meñique y pulgar
            x_pinky = int(landmarks[20].x * w)  # Punta del meñique
            y_pinky = int(landmarks[20].y * h)
            
            distance_pinky_thumb = np.hypot(x_pinky - x_thumb, y_pinky - y_thumb)

            cv2.circle(frame, (x_pinky, y_pinky), 3, (0, 255, 0), -1)  # Meñique
            cv2.line(frame, (x_pinky, y_pinky), (x_thumb, y_thumb), (0, 255, 0), 1)

            # Clic izquierdo (índice + pulgar)
            if distance_index_thumb < 40:
                if not click_down:
                    click_down = True
                    pyautogui.click()
                    cv2.putText(frame, 'L-Click!', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                click_down = False

            # Clic derecho (meñique + pulgar)
            if distance_pinky_thumb < 50:  # Distancia ligeramente mayor para el meñique
                if not right_click_down:
                    right_click_down = True
                    pyautogui.rightClick()
                    cv2.putText(frame, 'R-Click!', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            else:
                right_click_down = False

    # Redimensionar el frame para que se ajuste a la ventana pequeña
    frame_small = cv2.resize(frame, (200, 150))
    
    cv2.imshow(window_name, frame_small)
    
    # Mantener la ventana siempre en la posición correcta (por si se mueve)
    cv2.moveWindow(window_name, 0, 0)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()