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


# Per tumor
## Exp1 
### Model
Resnet10
### Input
per tumor with 20% bording for width and height, 5% slices. 
filter tumor size < 25% quantile

## Exp2
### Model
Resnet10
### Input
per tumor with bounding 64,64,32 if smaller than this bounding, otherwise keep original size.

## Exp3
### Model
Swin-TS
### Input
per tumor with bounding 64,64,32 if smaller than this bounding, otherwise keep original size. filter tumor size < 25 % quantile.

## Exp4 
### Model
Resnet10
### Input
Per tumor with pure HGPs Resnet

## Exp5
### Model
Swin-TS
### Input
per tumor with bounding 64,64,32, padding 64,64,32

## Exp6
Per tumor bounding 64,64,32

# Largest tumor
## Exp1
largest tumor
## Exp2 
### Model
Resnet10
### Input
largest+pure HGPs