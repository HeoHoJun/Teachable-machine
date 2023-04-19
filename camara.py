import cv2

# 카메라 켜기
cap = cv2.VideoCapture(0)

# 카메라가 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 프레임을 반복적으로 캡처
while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 프레임 읽기에 실패한 경우 종료
    if not ret:
        break

    # 프레임 출력
    cv2.imshow('Camera', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()