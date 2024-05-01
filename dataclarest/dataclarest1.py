import os
import cv2
import dlib
from PIL import Image

def resize_and_crop(img, modified_path, size, crop_type='middle', name='image.png'):
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    
    if ratio > img_ratio:
        img = img.resize((size[0], int(round(size[0] * img.size[1] / img.size[0]))), Image.LANCZOS)     
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, int(round((img.size[1] - size[1]) / 2)), img.size[0], int(round((img.size[1] + size[1]) / 2)))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((int(round(size[1] * img.size[0] / img.size[1])), size[1]), Image.LANCZOS)
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (int(round((img.size[0] - size[0]) / 2)), 0, int(round((img.size[0] + size[0]) / 2)), img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else:
        img = img.resize((size[0], size[1]), Image.LANCZOS)
        
    os.chdir(modified_path)
    img.save(name, "PNG")

def extract_head(image_path, shape_predictor_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 감지기 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    
    # 이미지에서 얼굴 감지
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # 감지된 얼굴에서 상단 좌표 추출
        top_left_x = face.left()
        top_left_y = face.top()
        
        # 머리 부분 추출 (얼굴 상단으로 가정)
        head_image = image[top_left_y:top_left_y+face.height(), top_left_x:top_left_x+face.width()]
        
        # 머리 이미지 반환
        return head_image

def process_images(input_path, output_path, size, shape_predictor_path):
    # 입력 경로에서 이미지 목록 가져오기
    files = os.listdir(input_path)
    
    # 각 이미지에 대해 처리 수행
    for file in files:
        # 이미지 경로
        image_path = os.path.join(input_path, file)
        
        # 머리 이미지 추출
        head_image = extract_head(image_path, shape_predictor_path)
        
        # 머리 이미지가 없는 경우 다음 이미지로 넘어감
        if head_image is None:
            continue
        
        # PIL 이미지로 변환
        pil_image = Image.fromarray(cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB))
        
        # 머리 이미지 크기 조정 및 저장
        resize_and_crop(pil_image, output_path, size, name=file)

# 입력 및 출력 경로 설정
input_path = r"C:\Users\Administrator\Desktop\dataclarest\input"
output_path = r"C:\Users\Administrator\Desktop\dataclarest\output"

# 이미지 크기 설정
size = (1024, 1024)

# 모양 예측기 파일 경로 설정
shape_predictor_path = r"C:\Users\Administrator\Desktop\dataclarest\shape_predictor_68_face_landmarks.dat"

# 이미지 처리 함수 호출
process_images(input_path, output_path, size, shape_predictor_path)