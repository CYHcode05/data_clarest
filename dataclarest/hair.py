import cv2
import numpy as np

# 이미지에서 얼굴 감지 함수
def detect_faces(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 감지기 초기화 (Haar cascade classifier 사용)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    return faces

# 이미지에서 얼굴 및 머리 데이터 추출 및 .dat 파일로 저장 함수
def extract_face_data(image_path, output_dat_path):
    faces = detect_faces(image_path)
    
    # .dat 파일 열기
    with open(output_dat_path, 'w') as f:
        # 각 얼굴에 대해 반복
        for i, (x, y, w, h) in enumerate(faces):
            face_image = cv2.imread(image_path)[y:y+h, x:x+w]
            # 이미지 크기를 조정하거나 다른 전처리를 수행할 수 있음
            # 예를 들어 face_image = cv2.resize(face_image, (100, 100))
            
            # 얼굴 이미지의 RGB 값 추출하여 .dat 파일에 저장
            for row in face_image:
                for pixel in row:
                    f.write(','.join(str(val) for val in pixel) + '\n')

# 이미지 경로와 .dat 파일 경로 설정
image_path = r'C:\Users\Administrator\Desktop\dataclarest\input'
output_dat_path = r'C:\Users\Administrator\Desktop\dataclarest\output'

# 얼굴 데이터 추출하여 .dat 파일로 저장
extract_face_data(image_path, output_dat_path)