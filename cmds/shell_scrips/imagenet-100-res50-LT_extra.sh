# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--epochs) pretrain_epochs="$2"; shift; shift ;;
    -s|--split) pretrain_split="$2"; shift; shift ;;
    -p|--port) port="$2"; shift; shift ;;
    -w|--workers) workers="$2"; shift; shift ;;
    -g|--GPU_NUM) GPU_NUM=("$2"); shift; shift ;;
    --data) data=("$2"); shift; shift ;;
    --seed) seed=("$2"); shift; shift ;;
    --temp) temp=("$2"); shift; shift ;;
    --pretrain_lr) pretrain_lr=("$2"); shift; shift ;;
    --pretrain_batch_size) pretrain_batch_size=("$2"); shift; shift ;;
    --only_pretraining) only_pretraining=("$2"); shift; shift ;;
    --only_finetuning) only_finetuning=("$2"); shift; shift ;;
    --test_only) test_only=("$2"); shift; shift ;;
    --only_few_shot) only_few_shot=("$2"); shift; shift ;;
    --few_shot_split) few_shot_split=("$2"); shift; shift ;;
    --linear_eval_seed) linear_eval_seed=("$2"); shift; shift ;;
    --save_dir) save_dir=("$2"); shift; shift ;;
    --online_sampling) online_sampling=("$2"); shift; shift ;;
    --resume) resume=("$2"); shift; shift ;;
    --sample_interval) sample_interval=("$2"); shift; shift ;;
    --warmup) warmup=("$2"); shift; shift ;;
    --k_largest_logits) k_largest_logits=("$2"); shift; shift ;;
    --k_means_clusters) k_means_clusters=("$2"); shift; shift ;;
    --budget) budget=("$2"); shift; shift ;;
    --author_dir) author_dir=("$2"); shift; shift ;;
    --sup_weight) sup_weight=("$2"); shift; shift ;;
    --cluster_temperature) cluster_temperature=("$2"); shift; shift ;;
    --sample_set) sample_set=("$2"); shift; shift ;;
    --COLT) COLT=("$2"); shift; shift ;;
    *) echo "${1} is not found"; exit 125;
esac
done


port=${port:-4833}
echo ${port}
pretrain_epochs=${pretrain_epochs:-500}
pretrain_split=${pretrain_split:-imageNet_100_LT_train}
workers=${workers:-10}
pretrain_batch_size=${pretrain_batch_size:-256}
linear_eval_seed=${linear_eval_seed:-1}
temp=${temp:-0.2}
pretrain_lr=${pretrain_lr:-0.5}
only_pretraining=${only_pretraining:-False}
only_finetuning=${only_finetuning:-False}
test_only=${test_only:-False}
only_few_shot=${only_few_shot:-False}
few_shot_split=${few_shot_split:-imageNet_100_sub_balance_train_0.01}
data=${data:-/ImageNet/}
seed=${seed:-1}
save_dir=${save_dir:-False}
resume=${resume:-False}
sample_interval=${sample_interval:--1}
warmup=${warmup:--1}
k_largest_logits=${k_largest_logits:-10}
k_means_clusters=${k_means_clusters:-10}
budget=${budget:-10000}
sup_weight=${sup_weight:-0.01}
cluster_temperature=${cluster_temperature:-1.0}
sample_set=${sample_set:-0}
COLT=${COLT:-False}
pretrain_name=${pretrain_split}_res50

cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} train_simCLR.py \
${pretrain_name} --epochs ${pretrain_epochs} \
--batch_size ${pretrain_batch_size} --output_ch 128 --lr ${pretrain_lr} --temperature ${temp} --model res50 \
--dataset imagenet-100 --imagenetCustomSplit ${pretrain_split} --save-dir ${save_dir} --optimizer sgd \
--num_workers ${workers} --seed ${seed} --data ${data} \
--sample_interval ${sample_interval} --warmup ${warmup} --k_largest_logits ${k_largest_logits} --k_means_clusters ${k_means_clusters} --budget ${budget} \
--sup_weight ${sup_weight} --cluster_temperature ${cluster_temperature} --sample_set ${sample_set}"

if [[ ${COLT} == "True" ]]
then
  cmd="${cmd} --COLT"
fi

if [[ ${resume} == "True" ]]
then
  cmd="${cmd} --resume"
fi

tuneLr=30
cmd_full="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} \
train_imagenet.py ${pretrain_name}_tune \
--decreasing_lr 10,20 --weight-decay 0 --epochs 30 --lr ${tuneLr} --batch-size 512 \
--model res50 --fullset --save-dir ${save_dir}_tune --dataset imagenet-100 \
--checkpoint ${save_dir}/${pretrain_name}/model_${pretrain_epochs}.pt \
--cvt_state_dict --world_size ${GPU_NUM} --port ${port} --num_workers ${workers} --test_freq 2 \
--seed ${linear_eval_seed} --data ${data}"

if [[ ${test_only} == "True" ]]
then
  cmd_full="${cmd_full} --test_only"
fi

tuneLr=30
cmd_few_shot="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} \
  train_imagenet.py ${pretrain_name}_few_shot \
  --decreasing_lr 40,60 --weight-decay 0 --epochs 100 --lr ${tuneLr} --batch-size 512 \
  --model res50 --save-dir ${save_dir}_tune --dataset imagenet-100 --customSplit ${few_shot_split} \
  --checkpoint ${save_dir}/${pretrain_name}/model_${pretrain_epochs}.pt \
  --cvt_state_dict --world_size ${GPU_NUM} --port ${port} --num_workers ${workers} --test_freq 10 --data ${data}"

if [[ ${test_only} == "True" ]]
then
  cmd_few_shot="${cmd_few_shot} --test_only"
fi

mkdir -p ${save_dir}/${pretrain_name}

if [[ ${only_few_shot} == "False" ]]
then
  if [[ ${only_finetuning} == "False" ]]
  then
    echo ${cmd} >> ${save_dir}/${pretrain_name}/bash_log.txt
    echo ${cmd}
    ${cmd}
  fi

  if [[ ${only_pretraining} == "False" ]]
  then
    echo ${cmd_full} >> ${save_dir}/${pretrain_name}/bash_log.txt
    echo ${cmd_full}
    ${cmd_full}

    echo ${cmd_few_shot} >> ${save_dir}/${pretrain_name}/bash_log.txt
    echo ${cmd_few_shot}
    ${cmd_few_shot}
  fi
else
  echo ${cmd_few_shot} >> ${save_dir}/${pretrain_name}/bash_log.txt
  echo ${cmd_few_shot}
  ${cmd_few_shot}
fi
