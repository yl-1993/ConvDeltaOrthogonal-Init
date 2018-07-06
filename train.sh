GPU=0

arch=resnet50
# arch=resnet101
# arch=resnet152
lr=0.1

# arch=van32
# lr=0.01

# conv_init='kaiming_normal'
conv_init='conv_delta_orthogonal'

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --lr $lr \
    --arch $arch \
    --conv-init $conv_init \
    2>&1 | tee log-$arch-$conv_init.txt
