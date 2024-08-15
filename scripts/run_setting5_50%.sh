tfpc=2
ratio='50%_labeled'
setting=setting_5
device='cuda:0'
time=$(date "+%Y_%m_%d_%H_%M_%S")
pre_train_epochs=50
main_train_epochs=10
# seg_weight=2.0
# temp_weight=0.5

echo "Time: $time"
echo "Setting: $setting"
echo "Device: $device"
echo "Pre-train epochs: $pre_train_epochs"
echo "Main train epochs: $main_train_epochs"    

python tools/$setting/pre_train.py \
    --t=$ratio --tfpc=$tfpc \
    --epochs=$pre_train_epochs \
    --device=$device \
    --seg-weight=1 \
    --temp-weight=2

python tools/$setting/main_train.py \
    --pre-train=best_eval_model.pth \
    --t=$ratio --tfpc=$tfpc \
    --epochs=$main_train_epochs \
    --device=$device \
    --seg-weight=2 \
    --temp-weight=0.1

python tools/$setting/test.py --t=$ratio --load='best_eval_model.pth'
python tools/$setting/eval.py --t=$ratio