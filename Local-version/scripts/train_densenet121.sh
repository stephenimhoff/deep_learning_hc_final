CUDA_VISIBLE_DEVICES=0 python train.py \
--pretrained \
--vision-backbone densenet121 \
--save-dir checkpoints \
--epochs 40 \
--lr 0.0001 \
--beta-rank 1 \
--beta-map 0.01 \
--beta-con 0.01 \
--neg-penalty 0.20 \
--textual-embeddings embeddings/nih_chest_xray_biobert.npy \
--data-root "C:/Users/Stephen Imhoff/OneDrive/UIUC MCS/Deep Learning for Healthcare/Final project/CXR-ML-GZSL"

#changing number of epocsh to 1 instead of 40 to fail faster