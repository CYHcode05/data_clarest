import dlib
import cv2
from skimage import io

# dlib의 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def apply_landmarks_to_image(source_image_path, target_image_path):
    # 이미지를 읽어옴
    source_img = io.imread(source_image_path)
    target_img = io.imread(target_image_path)
    
    # grayscale로 변환
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    source_faces = detector(source_gray)
    target_faces = detector(target_gray)

    # 각 얼굴에 대해 처리
    for source_face, target_face in zip(source_faces, target_faces):
        # 얼굴 랜드마크 예측
        source_landmarks = predictor(source_gray, source_face)
        target_landmarks = predictor(target_gray, target_face)

        # 랜드마크 좌표를 리스트로 저장
        source_landmarks_list = [(landmark.x, landmark.y) for landmark in source_landmarks.parts()]
        target_landmarks_list = [(landmark.x, landmark.y) for landmark in target_landmarks.parts()]

        # 랜드마크를 다른 얼굴에 적용
        for source_point, target_point in zip(source_landmarks_list, target_landmarks_list):
            # 두 점 사이의 거리 계산
            dx = target_point[0] - source_point[0]
            dy = target_point[1] - source_point[1]
            
            # 타겟 이미지에 랜드마크 그리기
            cv2.circle(target_img, (target_point[0] + dx, target_point[1] + dy), 1, (0, 0, 255), -1)
    
    # 결과 이미지를 출력
    cv2.imshow("Result", target_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 랜드마크를 적용할 이미지 경로 지정하여 함수 호출
apply_landmarks_to_image(r"C:\Users\Administrator\Desktop\dataclarest\face_with_landmarks.jpg", r"C:\Users\Administrator\Desktop\dataclarest\output\11.png")