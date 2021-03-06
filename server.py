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

# ?????nh ngh??a app = FastAPI()
app = FastAPI()

# trang ch???
@app.get('/')
def index():
    return {'message': 'API su dung cho app Quan ly chi tieu cua nhom 19521366 mon CS526'}

# api predict
@app.post("/predict")
def ocr(data: img_url):
    '''
    H??m n??y nh???n ?????u v??o l?? url c???a ???nh
    '''
    # Detect v??? tr?? th??ng tin tr??n h??a ????n
    score_thr = 0.85 # Ng?????ng 
    dict_detection = {} # Dict detection
    received = data.dict()
    img_url = received['url']
    im = Image.open(requests.get(img_url, stream=True).raw)
    image = np.array(im)
    result = inference_detector(model, image)
    
    # L???y bounding box

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
    
    lst_bboxes = [] # List ch???a c??c k???t qu??? detection
    
    # Th??m bounding box v??o detection list

    for cls, bbox in zip(labels, bboxes):
        lst_bboxes.append([int(cls), int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), float(bbox[4])])
    
    # X??? l?? OCR
    dict_per_cls = {0: [], 1: [], 2: [], 3: []} # Dict bbox sau khi chu???n h??a theo cls
    
    # Chu???n h??a bounding box (X???p l???i th??? t??? c???a bbox class ?????a ch???)
    for bbox in lst_bboxes:
        try:
            dict_per_cls[bbox[0]].append(bbox)
        except:
            pass

    dict_per_cls[1] = Sort(dict_per_cls[1])

    # Th???c hi???n OCR
    dict_cls = {0: [], 1: [], 2: [], 3: []} # Dict text theo class
    for cls in dict_per_cls:
        dict_cls[cls] = []
        for bbox in dict_per_cls[cls]:
            x1,y1,x2,y2 = bbox[1], bbox[2], bbox[3], bbox[4] # L???y ra t???a ?????
            crop_img = image[y1:y2,x1:x2] # C???t ra
            w, h = crop_img.shape[1], crop_img.shape[0] # L???y w, h c???a ph???n v???a c???t
            if w <= h:
              crop_img = cv2.rotate(crop_img, cv2.cv2.ROTATE_90_CLOCKWISE) # N???u ??ang b??? d???c th?? xoay l???i
            
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
            pil_image = Image.fromarray(rotated) # Convert t??? cv2 sang PIL
            text = model_ocr[cls].predict(pil_image) # D??? ??o??n text
            dict_cls[cls].append(text) # Th??m v??o k???t qu??? OCR
    
    # Chu???n h??a l???i k???t qu??? OCR
    totalcost = list(set(dict_cls[3])) # Lo???i c??c k???t qu??? t???ng ti???n gi???ng nhau
    if totalcost: # X??? l?? n???u c?? t???n t???i t???ng ti???n
        if checkInt(totalcost[0][0]) or (totalcost[0][0] == 'V'):
            totalcost = totalcost[::-1]
   
        if len(totalcost) > 1:
            totalcost = [totalcost[0], totalcost[-1].split(" ")[0]]
    
    # N???i l???i c??c k???t qu??? trong list
    
    try:
        seller = ' '.join(dict_cls[0])
    except:
        seller = 'Kh??ng th??? nh???n bi???t'

    try:
        address = ' '.join(dict_cls[1])
    except:
        address = 'Kh??ng th??? nh???n bi???t'

    try:
        timestamp = ' '.join(dict_cls[2])
    except:
        timestamp = 'Kh??ng th??? nh???n bi???t'
    
    try:
        totalcost = totalcost[-1]
    except:
        totalcost = 'Kh??ng th??? nh???n bi???t'

    # Tr??? v??? k???t qu??? ra json
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
