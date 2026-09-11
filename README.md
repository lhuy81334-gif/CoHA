# YOLOv8 Mask Detection

## Chay project

Dung Python he thong thay vi `.conda` trong thu muc nay, vi moi truong `.conda` hien dang bi loi thieu ca thu vien chuan Python.

```powershell
python -m pip install -r requirements.txt
```

Dat file model `best.pt` vao thu muc project, hoac truyen duong dan model bang `--model`.

Chay voi anh:

```powershell
python main.py --image image.png --output output_test.jpg
```

Chay voi video:

```powershell
python main.py --video test.mp4 --output output_video.avi
```

Chay voi webcam:

```powershell
python main.py --webcam 0
```

Chay webcam voi che do kiem tra nghiem hon:

```powershell
python main.py --webcam 0 --confirm-seconds 3 --mask-grace-seconds 1.5 --alert-seconds 0.8 --mask-conf 0.85 --max-skin-ratio-for-mask 0.35
```

Khi da accept va bat dau monitoring, neu thay `no_mask`, `no_mask_visible_face`, `uncertain`, `uncertain_cover` hoac `no_face`, chuong trinh se luu frame vao:

```text
human_check/
```

Nhan `q` de tat cua so khi chay video hoac webcam.

## Train lai model

File `data.yaml` dang dung duong dan dataset tuong doi:

```yaml
path: dataset
```

Neu muon train, hay tao dung cau truc:

```text
data/
  train/images
  train/labels
  test/images
  test/labels
```

Sau do co the bo comment va chay lenh train tu Python hoac dung YOLO CLI.
