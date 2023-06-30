TRAIN_EPOCHS=70
TRAIN_LR=1e-4
BATCH_SIZE=128
SEED=0

JOB_NAME=training_pdm_open_model
CACHE_PATH=/path/to/cache
USE_CACHE_WITHOUT_DATASET=False

python $NUPLAN_DEVKIT_ROOT/nuplan-devkit/nuplan/planning/script/run_training.py \
seed=$SEED \
+training=training_pdm_open_model \
job_name=$JOB_NAME \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
data_loader.params.batch_size=$BATCH_SIZE \
optimizer.lr=$TRAIN_LR \
hydra.searchpath="[pkg://nuplan_garage.planning.script.config.common, pkg://nuplan_garage.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"