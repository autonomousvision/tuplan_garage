TRAIN_EPOCHS=100
TRAIN_LR=1e-4
TRAIN_LR_MILESTONES=[50,75]
TRAIN_LR_DECAY=0.1
BATCH_SIZE=64
SEED=0

JOB_NAME=training_pdm_offset_model
CACHE_PATH=/path/to/cache/
USE_CACHE_WITHOUT_DATASET=False

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_training.py \
seed=$SEED \
py_func=train \
+training=training_pdm_offset_model \
job_name=$JOB_NAME \
scenario_builder=nuplan \
cache.cache_path=$CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
lightning.trainer.params.max_epochs=$TRAIN_EPOCHS \
data_loader.params.batch_size=$BATCH_SIZE \
optimizer.lr=$TRAIN_LR \
lr_scheduler=multistep_lr \
lr_scheduler.milestones=$TRAIN_LR_MILESTONES \
lr_scheduler.gamma=$TRAIN_LR_DECAY \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.training, pkg://tuplan_garage.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
