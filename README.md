# Vehicle detection dataset

1/ modify config at `\Co-DETR\projects\configs\_base_\datasets\vehicle_detection.py`

2/ modify config of model at `\Co-DETR\projects\configs\` (see Model readme at Co-DETR)

**for example:** model Co-DINO	R50	12	DETR	COCO	52.1 => modify file `\Co-DETR\projects\configs\co_dino\co_dino_5scale_r50_1x_coco.py`

3/ modify the checkpoint path (line 20):
- in docker: /co-detr/data/pretrained_models/co_dino_5scale_r50_1x_coco.pth
- in kaggle: /kaggle/Vehicle-Detection/Co-DETR/data/pretrained_models/co_dino_5scale_r50_1x_coco.pth
3/ run train detector

```bash
bash tools/dist_train.sh projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py 2 outputs
```