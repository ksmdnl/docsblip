devices=[2]

model=src.models.docsblip_stage1.DocsBlip
task_name=docsblip_stage1
tags=[docvqa,docsblip,stage1]

ref_lr=1e-4
start_lr=1e-6
final_lr=1e-5
ref_wd=0.05
warmup_steps=2000

python src/train.py \
    +matmul_precision=medium \
    trainer.devices=${devices}\
    trainer.max_epochs=40 \
    +trainer.precision=bf16-mixed \
    +trainer.num_sanity_val_steps=0 \
    trainer=gpu \
    model=docsblip \
    model._target_=${model} \
    model.ref_lr=${ref_lr} \
    model.start_lr=${start_lr} \
    model.final_lr=${final_lr} \
    model.ref_wd=${ref_wd} \
    model.warmup_steps=${warmup_steps} \
    experiment=docsblip \
    data=docvqa \
    +data.num_workers=8 \
    +data.batch_size=128 \
    callbacks.rich_progress_bar=null \
    +logger.wandb.name=${task_name}\
    logger.wandb.tags=${tags}\
    task_name=${task_name}\
    # +data.subset_size=200 \
    # logger.wandb=null