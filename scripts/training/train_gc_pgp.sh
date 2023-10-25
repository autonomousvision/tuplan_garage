BATCH_SIZE=32
SEED=0

PRETRAIN_EPOCHS=20
PRETRAIN_LR=1e-4
TRAIN_EPOCHS=90
TRAIN_LR=1e-4
TRAIN_LR_MILESTONES=[40,50,55]
TRAIN_LR_DECAY=0.5

JOB_NAME=training_gc_pgp_model
CACHE_PATH=/path/to/cache/
USE_CACHE_WITHOUT_DATASET=False

ROUTE_FEATURE=FALSE
ROUTE_MASK=FALSE
HARD_MASK=FALSE
TRAFFIC_LIGHT=TRUE

echo "Starting Pre-Training with gt traversals as input for decoder"
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_training.py \
seed=$SEED \
py_func=train \
+training=training_gc_pgp_model \
job_name=$JOB_NAME \
scenario_builder=nuplan \
scenario_filter.num_scenarios_per_type=4000 \
cache.cache_path=$CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
callbacks.visualization_callback.pixel_size=0.25 \
callbacks.multimodal_visualization_callback.pixel_size=0.25 \
lightning.trainer.params.max_epochs=$PRETRAIN_EPOCHS \
lightning.trainer.params.max_time=null \
data_loader.params.batch_size=$BATCH_SIZE \
optimizer.lr=$PRETRAIN_LR \
lr_scheduler=multistep_lr \
lr_scheduler.milestones=$TRAIN_LR_MILESTONES \
lr_scheduler.gamma=$TRAIN_LR_DECAY \
model.encoder.use_red_light_feature=$TRAFFIC_LIGHT \
model.aggregator.use_route_mask=$ROUTE_MASK \
model.aggregator.hard_masking=$HARD_MASK \
model.aggregator.pre_train=true \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.training, pkg://tuplan_garage.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"

echo "Starting Training with aggregator traversals as input for decoder"
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_training.py \
seed=$SEED \
py_func=train \
+training=training_gc_pgp_model \
job_name=$JOB_NAME \
scenario_builder=nuplan \
scenario_filter.num_scenarios_per_type=4000 \
cache.cache_path=$CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
callbacks.visualization_callback.pixel_size=0.25 \
callbacks.multimodal_visualization_callback.pixel_size=0.25 \
lightning.trainer.params.max_epochs=$TRAIN_EPOCHS \
lightning.trainer.params.max_time=null \
lightning.trainer.checkpoint.resume_training=true \
data_loader.params.batch_size=$BATCH_SIZE \
optimizer.lr=$TRAIN_LR \
lr_scheduler=multistep_lr \
model.encoder.use_red_light_feature=$TRAFFIC_LIGHT \
model.aggregator.use_route_mask=$ROUTE_MASK \
model.aggregator.hard_masking=$HARD_MASK \
model.aggregator.pre_train=false \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.training, pkg://tuplan_garage.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
