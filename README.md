Source server sử dụng cho đồ án quản lý chi tiêu

## Cài thư viện cần thiết

```
git clone https://github.com/open-mmlab/mmdetection
!pip install colabcode
!pip install pydantic
!pip install fastapi
%cd mmdetection
!pip install -r requirements/build.txt
!pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
!pip install -v -e .  # or "python setup.py develop"
!pip install mmcv-full==latest+torch1.7.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
!pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
!pip install --quiet vietocr==0.3.2
%cd ..

```

## Chạy server

```
python server.py
```
