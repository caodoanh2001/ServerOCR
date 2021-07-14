from pydantic import BaseModel
import numpy as np
from utils import *
from fastapi import FastAPI
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from PIL import Image
import requests
import pdb

# import colabcode
from colabcode import ColabCode

# import mmdetection
from mmdet.apis import init_detector, inference_detector

# import vietocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# import model vietocr

config_seller = Cfg.load_config_from_name('vgg_transformer')
config_seller['weights'] = './models/OCR/seller.pth'
config_seller['cnn']['pretrained']=False
config_seller['device'] = 'cuda:0'
config_seller['predictor']['beamsearch']=False
detector_seller = Predictor(config_seller)

config_address = Cfg.load_config_from_name('vgg_transformer')
config_address['weights'] = './models/OCR/address.pth'
config_address['cnn']['pretrained']=False
config_address['device'] = 'cuda:0'
config_address['predictor']['beamsearch']=False
detector_address = Predictor(config_address)

config_timestamp = Cfg.load_config_from_name('vgg_transformer')
config_timestamp['weights'] = './models/OCR/timestamp.pth'
config_timestamp['cnn']['pretrained']=False
config_timestamp['device'] = 'cuda:0'
config_timestamp['predictor']['beamsearch']=False
detector_timestamp = Predictor(config_timestamp)

config_totalcost = Cfg.load_config_from_name('vgg_transformer')
config_totalcost['weights'] = './models/OCR/totalcost.pth'
config_totalcost['cnn']['pretrained']=False
config_totalcost['device'] = 'cuda:0'
config_totalcost['predictor']['beamsearch']=False
detector_totalcost = Predictor(config_totalcost)

# import checkpoint detection

checkpoint_detection= './models/detection/epoch_18.pth'
config_detection = './models/detection/config101_62_1x.py'
model = init_detector(config_detection, checkpoint_detection, device='cuda:0')

# Dict model and dict cls

model_ocr = {
    0: detector_seller,
    1: detector_address,
    2: detector_timestamp,
    3: detector_totalcost
}

class img_url(BaseModel):
    url: str 
    class Config:
        schema_extra = {
            "example": {
                "url": "https://i.imgur.com/2BFKh2k.jpg", 
            }
        }

def Sort(sub_li):
    return(sorted(sub_li, key = lambda x: int(float(x[2]))))  

# Định nghĩa app = FastAPI()
app = FastAPI()

# trang chủ
@app.get('/')
def index():
    return {'message': 'API su dung cho app Quan ly chi tieu cua nhom 19521366 mon CS526'}

# api predict
@app.post("/predict")
def ocr(data: img_url):
    '''
    Hàm này nhận đầu vào là url của ảnh
    '''
    # Detect vị trí thông tin trên hóa đơn
    score_thr = 0.85 # Ngưỡng 
    dict_detection = {} # Dict detection
    received = data.dict()
    img_url = received['url']
    im = Image.open(requests.get(img_url, stream=True).raw)
    image = np.array(im)
    result = inference_detector(model, image)
    
    # Lấy bounding box

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    
    lst_bboxes = [] # List chứa các kết quả detection
    
    # Thêm bounding box vào detection list

    for cls, bbox in zip(labels, bboxes):
        lst_bboxes.append([int(cls), int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), float(bbox[4])])
    
    # Xử lý OCR
    dict_per_cls = {0: [], 1: [], 2: [], 3: []} # Dict bbox sau khi chuẩn hóa theo cls
    
    # Chuẩn hóa bounding box (Xếp lại thứ tự của bbox class địa chỉ)
    for bbox in lst_bboxes:
        try:
            dict_per_cls[bbox[0]].append(bbox)
        except:
            pass

    dict_per_cls[1] = Sort(dict_per_cls[1])

    # Thực hiện OCR
    dict_cls = {0: [], 1: [], 2: [], 3: []} # Dict text theo class
    for cls in dict_per_cls:
        dict_cls[cls] = []
        for bbox in dict_per_cls[cls]:
            x1,y1,x2,y2 = bbox[1], bbox[2], bbox[3], bbox[4] # Lấy ra tọa độ
            crop_img = image[y1:y2,x1:x2] # Cắt ra
            w, h = crop_img.shape[1], crop_img.shape[0] # Lấy w, h của phần vừa cắt
            if w <= h:
              crop_img = cv2.rotate(crop_img, cv2.cv2.ROTATE_90_CLOCKWISE) # Nếu đang bị dọc thì xoay lại
            
            # PREPROCESSING
            _, skewd = correct_skew(crop_img)
            gray = cv2.cvtColor(skewd, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # threshold the image, setting all foreground pixels to 255 and all background pixels to 0
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # grab the (x, y) coordinates of all pixel values that are greater than zero, then use these coordinates to compute a rotated bounding box that contains all coordinates
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]

            # the `cv2.minAreaRect` function returns values in the range [-90, 0); as the rectangle rotates clockwise the returned angle trends to 0 -- in this special case we need to add 90 degrees to the angle
            if angle < -45:
                angle = -(90 + angle)
            
            # otherwise, just take the inverse of the angle to make it positive
            else:
                angle = -angle

            # rotate the image to deskew it
            (h, w) = skewd.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(skewd, M, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rotated) # Convert từ cv2 sang PIL
            text = model_ocr[cls].predict(pil_image) # Dự đoán text
            dict_cls[cls].append(text) # Thêm vào kết quả OCR
    
    # Chuẩn hóa lại kết quả OCR
    totalcost = list(set(dict_cls[3])) # Loại các kết quả tổng tiền giống nhau
    if totalcost: # Xử lý nếu có tồn tại tổng tiền
        if checkInt(totalcost[0][0]) or (totalcost[0][0] == 'V'):
            totalcost = totalcost[::-1]
   
        if len(totalcost) > 1:
            totalcost = [totalcost[0], totalcost[-1].split(" ")[0]]
    
    # Nối lại các kết quả trong list
    
    try:
        seller = ' '.join(dict_cls[0])
    except:
        seller = 'Không thể nhận biết'

    try:
        address = ' '.join(dict_cls[1])
    except:
        address = 'Không thể nhận biết'

    try:
        timestamp = ' '.join(dict_cls[2])
    except:
        timestamp = 'Không thể nhận biết'
    
    try:
        totalcost = totalcost[-1]
    except:
        totalcost = 'Không thể nhận biết'

    # Trả về kết quả ra json
    dict_out = {
        "seller": seller,
        "address": address,
        "timestamp": timestamp,
        "totalcost": totalcost,
    }

    return dict_out

# run server
server = ColabCode(port=10001, code=False, authtoken = '1tZN1OIxC54y7we1FQlNRXusGcS_4vwyHgHDwhUi9vTKsmRKc')
server.run_app(app=app)
