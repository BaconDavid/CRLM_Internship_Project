# Experi
## Exp1
### Data
Resample to 0.7,0.7,1 (first crop then resample)
Largest tumor with liver slices+20 extra slices(fit the cluster, and some liver slices are small)
Window imageing
data aug
### Model
1. Resnet10: Only DA
    fold: 01,04,05
    No Resize and crop

2. Swin_trans:
    fold: 01,04,05
    Padding: 256,256,64
    CenterCrop: 256,256,64

## Exp2

Resample to 0.7,0.7,1 (first crop then resample)
Largest tumor with liver slices+20 extra slices(fit the cluster, and some liver slices are small)
window imageing
DA
**Resize and crop**  resize to 256,256,-1, crop to (256,256,64)

### Model
1. Resnet 
2. Swin_trans

## Exp3

Resample to 0.7,0.7,1 (first crop then resample)
Largest tumor with liver slices+20 extra slices(fit the cluster, and some liver slices are small)
No window imaging
Data Aug


## Exp4
