import cv2
import mediapipe as mp
import math
from collections import deque, Counter


cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


IMAGE_PATHS = {
    "chill": "images/CatChill.jpg",
    "haha":  "images/CatHaha.jpg",
    "like":  "images/CatLike.jpg",
    "tss":   "images/CatTss.jpg",
    "rock":  "images/DogRock.jpg",
}

TARGET = (200, 200)

images = {}
for name, path in IMAGE_PATHS.items():
    img = cv2.imread(path)
    if img is None:
        print(f"ВНИМАНИЕ: не загрузилась {path}")
    else:
        images[name] = cv2.resize(img, TARGET)


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def get_finger_states(hand):
    states = []
    wrist = hand.landmark[0]
    tip4  = hand.landmark[4]
    pip2  = hand.landmark[2]
    states.append(dist(tip4, wrist) > dist(pip2, wrist))
    for t, p in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        states.append(hand.landmark[t].y < hand.landmark[p].y)
    return states


def classify_gesture(hand):
    b, i, m, r, p = get_finger_states(hand)

    if i and not m and not r and not p:
        
        wrist = hand.landmark[0]
        tip8  = hand.landmark[8]
        pip5  = hand.landmark[5]
        if dist(tip8, wrist) < dist(pip5, wrist):
            return "haha"
        return "tss"

   
    if i and p and not m and not r:
        return "rock"

    if b and p and not i and not m and not r:
        return "chill"

    
    if b and not i and not m and not r and not p:
        return "like"

    return "unknown"


gesture_history = deque(maxlen=12)

def smooth_gesture(g):
    gesture_history.append(g)
    return Counter(gesture_history).most_common(1)[0][0]


while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    raw_gesture = "unknown"

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            raw_gesture = classify_gesture(hand)

    gesture = smooth_gesture(raw_gesture)

    cv2.putText(frame, gesture, (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    img = images.get(gesture)
    if img is not None:
        h, w = img.shape[:2]
        frame[10:10+h, 10:10+w] = img

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

hands.close()
cam.release()
cv2.destroyAllWindows()