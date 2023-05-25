set -e
set -x

# train the MECPformer
python train.py --weights ./weights/Conformer_small_patch16.pth

# infer the MECPformer
python infer.py --weights ./model_MECPformer/MECPformer_0917_6.pth
python evaluation.py --predict_dir save/out_cam --comment ./comment

# the ACM module
python ACM.py
