# prompt-DA
This is the 3DUNet version prompt-DA for infant tissue segmentation.
![image](https://github.com/MurasakiLin/prompt-DA/assets/127721194/96e03614-12fd-4660-8107-8078b64be37d)

## Prpare data
```
python prepare_hdf5_cutedge.py 
python hdf5_for_unlabel.py
```

## Training(After add dataset path in the code)
```
python train_base.py           #baseline
python train_onlyprompt.py     #only prompt
python train_adv.py            #only adversarial train
python train_comb.py           #combine prompt and adv
```
