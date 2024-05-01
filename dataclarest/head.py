import dlib
import cv2

# dlib의 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()

def resize_face_area(image_path1, image_path2, output_path):
    # 이미지 1 로드
    img1 = cv2.imread(image_path1)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 이미지 2 로드
    img2 = cv2.imread(image_path2)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 이미지 1에서 얼굴 감지
    faces1 = detector(gray1)
    if len(faces1) == 0:
        print("No faces detected in the first image.")
        return
    face1 = faces1[0]  # 첫 번째 얼굴만 고려

    # 이미지 2에서 얼굴 감지
    faces2 = detector(gray2)
    if len(faces2) == 0:
        print("No faces detected in the second image.")
        return
    face2 = faces2[0]  # 첫 번째 얼굴만 고려

    # 이미지 1의 얼굴 중심 좌표와 크기 계산
    center_x1 = (face1.left() + face1.right()) // 2
    center_y1 = (face1.top() + face1.bottom()) // 2
    face_width1 = face1.right() - face1.left()
    face_height1 = face1.bottom() - face1.top()

    # 이미지 2의 얼굴 중심 좌표와 크기 계산
    center_x2 = (face2.left() + face2.right()) // 2
    center_y2 = (face2.top() + face2.bottom()) // 2
    face_width2 = face2.right() - face2.left()
    face_height2 = face2.bottom() - face2.top()

    # 이미지 1과 이미지 2의 얼굴 크기 비교
    ratio = min(face_width1 / face_width2, face_height1 / face_height2)

    # 이미지 1과 이미지 2의 얼굴 주변 영역 크기 조정
    resized_face_width2 = int(face_width2 * ratio)
    resized_face_height2 = int(face_height2 * ratio)

    # 이미지 2의 얼굴 주변 영역 크기 조정
    left2 = max(0, center_x2 - resized_face_width2 // 2)
    top2 = max(0, center_y2 - resized_face_height2 // 2)
    right2 = min(img2.shape[1], center_x2 + resized_face_width2 // 2)
    bottom2 = min(img2.shape[0], center_y2 + resized_face_height2 // 2)

    # 아래로 이동하는 양 조정 (예: 아래로 50픽셀)
    shift_amount = -50
    top2 += shift_amount
    bottom2 += shift_amount

    # 이미지 2의 얼굴 주변 영역 크기 조정
    resized_face_area2 = img2[top2:bottom2, left2:right2]

    # 1024x1024 크기로 이미지 조정
    resized_image = cv2.resize(resized_face_area2, (1024, 1024))

    # 결과 이미지 저장
    cv2.imwrite(output_path, resized_image)

    print("Resized face area saved to:", output_path)

# 이미지 파일 경로 및 저장 경로 설정하여 함수 호출
resize_face_area("output/0.png", "output/15.PNG", "output/resized_face_area.png")
