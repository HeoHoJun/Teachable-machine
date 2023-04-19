import tensorflow.keras                                                                                                                                # 필요한 라이브러리를 가져온다.
import numpy as np
import cv2

model = tensorflow.keras.models.load_model('keras_model.h5')                                                                                           # 저장된 Keras 모델을 불러온다.

cap = cv2.VideoCapture(0)                                                                                                                              # 비디오 입력을 위해 웹캠 장치를 연다.

size = (224, 224)                                                                                                                                      # 모델에 입력될 이미지 크기를 정의한다.

classes = ['smartphone', 'cup', 'glass case']                                                                                                          # 모델이 인식할 클래스들을 정의한다.

while cap.isOpened():                                                                                                                                  # 웹캠으로부터 프레임을 계속해서 가져와서 처리한다.
    ret, img = cap.read()                                                                                                                              # 웹캠에서 프레임을 읽는다.
    if not ret:
        break

    h, w, _ = img.shape                                                                                                                                # 이미지의 높이와 너비를 가져온다.             
    cx = h / 2                                                                                                                                         # 이미지를 수직 축을 기준으로 중앙에 위치한 정사각형으로 자른다.
    img = img[:, 200:200+img.shape[0]]
    img = cv2.flip(img, 1)                                                                                                                             # 이미지를 수평으로 뒤집는다.

    img_input = cv2.resize(img, size)                                                                                                                  # 모델에 입력될 이미지를ㄹ 크기에 맞게 조절하고 전처리한다.
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = (img_input.astype(np.float32) / 127.0) - 1
    img_input = np.expand_dims(img_input, axis=0)

    prediction = model.predict(img_input)                                                                                                              # 모델을 사용하여 입력 이미지의 클래스를 예측한다.
    idx = np.argmax(prediction)

    cv2.putText(img, text=classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)            # 예측된 클래스 이름을 이미지 위에 표시한다.

    cv2.imshow('result', img)                                                                                                                          # 결과 이미지를 보여준다.
    if cv2.waitKey(1) == ord('q'):                                                                                                                     # 사용자가 'q' 키를 누르면 루프를 종료한다.
        break