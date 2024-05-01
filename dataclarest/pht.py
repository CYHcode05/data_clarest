import dlib
import cv2
from skimage import io

# dlib의 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_face_and_head_images(image_path):
    # 이미지를 읽어옴
    img = io.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = detector(gray)
    
    # 각 얼굴에 대해 처리
    for i, face in enumerate(faces):
        # 얼굴 랜드마크 예측
        landmarks = predictor(gray, face)

        # 랜드마크 좌표를 리스트로 저장
        landmarks_list = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_list.append((x, y))
        
        # 머리 영역의 좌표 계산
        top_x, top_y = min(landmarks_list, key=lambda x: x[1])
        bottom_x, bottom_y = landmarks_list[8]  # 아래턱 좌표 사용
        left_x, left_y = landmarks_list[0]
        right_x, right_y = landmarks_list[16]

        # 머리 영역 자르기
        head_image = img[top_y:bottom_y, left_x:right_x]

        # 얼굴 이미지에 랜드마크 그리기
        for landmark in landmarks_list:
            cv2.circle(img, landmark, 1, (0, 0, 255), -1)
        
        # 얼굴 이미지에 사각형 그리기
        cv2.rectangle(img, (left_x, top_y), (right_x, bottom_y), (0, 255, 0), 2)

        # 자른 머리 이미지와 얼굴 이미지를 파일로 저장
        cv2.imwrite(f"head_{i}.jpg", head_image)
        cv2.imwrite(f"face_{i}.jpg", img)

    # 전체 이미지에 대한 얼굴 및 랜드마크가 표시된 이미지도 저장
    cv2.imwrite("face_with_landmarks.jpg", img)

# 이미지 경로 지정하여 함수 호출
extract_face_and_head_images(r"C:\Users\Administrator\Desktop\dataclarest\output\0.png")