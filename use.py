import os

os.system('python train_cirkd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --data C:/dataset/cityscape/ \
    --save-dir C:/paper_codes/CIRKD/output/ckpt/ \
    --log-dir C:/paper_codes/CIRKD/output/log/ \
    --teacher-pretrained C:/paper_codes/CIRKD/ckpt_pre/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base C:/paper_codes/CIRKD/ckpt_pre/resnet18-imagenet.pth')

# os.system('python test.py \
#           --data C:/dataset/cityscape/ \
#           --pretrained C:/paper_codes/CIRKD/ckpt_pre/kd_deeplabv3_resnet18_citys_best_model.pth')


# os.system('python train_baseline.py \
#           --data C:/dataset/cityscape/ \
#           --pretrained-base C:/paper_codes/CIRKD/ckpt_pre/resnet18-imagenet.pth')