s=$(cat "$0")
iters=20000000
b=100
eval_freq=1000
cuda_device=2
iter_record_freq=0
activation=relu
base_folder=results/
model=fcn
width=2000
depth=5
dataset=cifar10
val_split=val
val_criterion=loss
val_best_k_models=1
val_patience=5000
val_ratio=0.1
val_best=--val_keep_best_model
datatext=$(echo "$dataset" | sed 's/digit_original_//')
datatext=$(echo "$datatext" | sed 's/doublemnist_lo_left_/dmnist_/')
min_iters=10000
lr=0.001
activation=relu
wd=0.01
for seed in 0
do
for layer_rank in -1 #10 15
do
lr_text=${lr//[ ]/_}
tag=dlayer_rank"$layer_rank"_"$model"_w"$width"d"$depth"_"$datatext"_lr"$lr_text"_s"$seed"
tag=$(echo "$tag" | sed 's/w0d0_//')
echo $tag 
python -u main.py --layer_rank $layer_rank --lr_algorithm adam --batch_conv --activation $activation --min_iters $min_iters --wd $wd --val_split $val_split --val_criterion $val_criterion --val_best_k_models $val_best_k_models --val_patience $val_patience --val_ratio $val_ratio $val_best  --be 100 -i --activation $activation --iter_record_freq $iter_record_freq --model $model --dataset $dataset --lr $lr --batch_size_train $b --iterations $iters --save_dir "$base_folder""$tag" --width $width --depth $depth --eval_freq $eval_freq --cm none  --cuda_device $cuda_device -s "$s" --seed $seed #> logs/$tag 2>&1 &
echo $!
sleep 0.1
done
done
wait
