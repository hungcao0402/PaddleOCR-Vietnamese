# PaddleOCR-Vietnamese
Scene text vietnamese

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
# Build docker image
```bash
docker build -t sast_srn .
```
Run docker image
```bash
docker run -v test_data:/data/test_data:ro submission_output:/data/submission_output sast_srn /bin/bash run.sh
```


