# PaddleOCR-Vietnamese
Scene text vietnamese

Link blog: http://tutorials.aiclub.cs.uit.edu.vn/index.php/2022/04/20/nhan-dang-chu-tieng-viet-trong-anh-ngoai-canh/
# Setup
```bash
pip install -r requirements.txt
pip install paddlepaddle-gpu
```
## Train detection model
```bash
python3 tools/train.py -c ./configs/det/SAST.yml
```
## Train recognition model
```bash
python3 tools/train.py -c ./configs/rec/SRN.yml
```

## Evaluation detection
```bash
python3 tools/eval.py -c ./configs/det/SAST.yml
```
## Evaluation recognition 
```bash
python3 tools/eval.py -c ./configs/rec/SRN.yml
```

## Predict detection
```bash
python3 tools/infer_det.py -c ./configs/det/SAST.yml -o Global.infer_img= #path_to_image
```
## Predict recognition
```bash
python3 tools/infer_rec.py -c ./configs/rec/SRN.yml -o Global.infer_img=im0001_1.jpg
```
# Convert to inference Model
```bash
python3 tools/export_model.py -c ./configs/det/SAST.yml  
python3 tools/export_model.py -c ./configs/rec/SRN.yml
```
# Detection and recognition concatenate 
```bash
python3 /content/drive/MyDrive/PaddleOCR/PaddleOCR/tools/infer/predict_system.py 
                --use_gpu=True \
                --det_algorithm="SAST" \
                --det_model_dir="./inference/SAST" \
                --rec_algorithm="SRN" \
                --rec_model_dir="./inference/SRN/" \
                --rec_image_shape="1, 64, 256" \
                --image_dir=#path_img \
                --rec_char_type="ch" \
                --drop_score=0.7 \
                --rec_char_dict_path="./ppocr/utils/dict/vi_vietnam.txt"
```
# Build docker image
```bash
docker build -t sast_srn .
```
Run docker image
```bash
docker run -v test_data:/data/test_data:ro submission_output:/data/submission_output sast_srn /bin/bash run.sh
```


