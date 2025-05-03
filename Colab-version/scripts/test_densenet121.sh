CUDA_VISIBLE_DEVICES=0 python drive/MyDrive/Colab\ Notebooks/test.py \
--vision-backbone densenet121 \
--textual-embeddings /content/drive/MyDrive/Colab\ Notebooks/embeddings/nih_chest_xray_biobert.npy \
--load-from /content/drive/MyDrive/Colab\ Notebooks/checkpoints/best_auroc_checkpoint.pth.tar
