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
    --lr) pretrain_lr=("$2"); shift; shift ;;
    --only_finetuning) only_finetuning=("$2"); shift; shift ;;
    --test_only) test_only=("$2"); shift; shift ;;
    --batch_size) batch_size=("$2"); shift; shift ;;
    --temp) pretrain_temp=("$2"); shift; shift ;;
    --few_shot_only) few_shot_only=("$2"); shift; shift ;;
    --few_shot_lr) few_shot_lr=("$2"); shift; shift ;;
    --few_shot_epochs) few_shot_epochs=("$2"); shift; shift ;;
    --save_dir) save_dir=("$2"); shift; shift ;;
    --resume) resume=("$2"); shift; shift ;;
    --COLT) COLT=("$2"); shift; shift ;;
    --data_root) data_root=("$2"); shift; shift ;;
    --sup_weight) sup_weight=("$2"); shift; shift ;;
    *) echo "${1} is not found"; exit 125;
esac
done

GPU_NUM=${GPU_NUM:-1}
workers=${workers:-5}
batch_size=${batch_size:-512}
few_shot_only=${few_shot_only:-False}
seed=10

port=${port:-4833}
pretrain_epochs=${pretrain_epochs:-2000}
pretrain_split=${pretrain_split:-cifar100_split1_D_i}
pretrain_lr=${pretrain_lr:-0.5}
pretrain_temp=${pretrain_temp:-0.2}
few_shot_lr=${few_shot_lr:-0.02}
few_shot_epochs=${few_shot_epochs:-300}
only_finetuning=${only_finetuning:-False}
test_only=${test_only:-False}
save_dir=${save_dir:-False}
COLT=${COLT:-False}
data_root=${data_root:-'path/to/cifar'}
sup_weight=${sup_weight:-0.2}
pretrain_name=res18_cifar100

cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} train_simCLR.py ${pretrain_name} --epochs ${pretrain_epochs} \
--batch_size ${batch_size} --optimizer sgd --lr ${pretrain_lr} --temperature ${pretrain_temp} --model res18 \
--trainSplit cifar100_imbSub_with_subsets/${pretrain_split}.npy --save-dir ${save_dir} --seed ${seed} \
--dataset cifar100 --output_ch 128 --num_workers ${workers} --data ${data_root} --sup_weight ${sup_weight}"

if [[ ${COLT} == "True" ]]
then
  cmd="${cmd} --COLT"
fi

if [[ ${resume} == "True" ]]
then
  cmd="${cmd} --resume"
fi

tuneLr=30
cmd_full="python train_cifar.py finetune \
--fixbn --wd 0 --model res18 --epochs 30 --lr ${tuneLr} --decreasing_lr 10,20 \
--trainSplit cifar100/cifar100_trainIdxList.npy --fixto layer4  --checkpoint ${save_dir}/${pretrain_name}/model_${pretrain_epochs}.pt \
--cvt_state_dict --save-dir ${save_dir}_tune --valiSplit cifar100/cifar100_valIdxList.npy --dataset cifar100 --data ${data_root}"

if [[ ${test_only} == "True" ]]
then
  cmd_full="${cmd_full} --test_only"
fi

tuneLr=30
index_split="$(echo ${pretrain_split} | grep -P 'split\K([0-9])' -o)"
train_split1=cifar100_split${index_split}_S_b
cmd_few_shot="python train_cifar.py few_shot \
--fixbn --wd 0 --model res18 --epochs 100 --lr ${tuneLr} --decreasing_lr 40,60 \
--trainSplit cifar100_imbSub_with_subsets/${train_split1}.npy --fixto layer4 --checkpoint ${save_dir}/${pretrain_name}/model_${pretrain_epochs}.pt \
--cvt_state_dict --save-dir ${save_dir}_tune --valiSplit cifar100/cifar100_valIdxList.npy --dataset cifar100 --test_freq 5 --data ${data_root}"

if [[ ${test_only} == "True" ]]
then
  cmd_few_shot="${cmd_few_shot} --test_only"
fi


if [ ${few_shot_only} == "False" ]
then
  mkdir -p ${save_dir}/${pretrain_name}

  if [ ${only_finetuning} == "False" ]
  then
    echo ${cmd} >> ${save_dir}/${pretrain_name}/bash_log.txt
    echo ${cmd}
    ${cmd}
  fi

  echo ${cmd_full} >> ${save_dir}/${pretrain_name}/bash_log.txt
  echo ${cmd_full}
  ${cmd_full}

  echo ${cmd_few_shot} >> ${save_dir}/${pretrain_name}/bash_log.txt
  echo ${cmd_few_shot}
  ${cmd_few_shot}
else
  echo ${cmd_few_shot} >> ${save_dir}/${pretrain_name}/bash_log.txt
  echo ${cmd_few_shot}
  ${cmd_few_shot}
fi
