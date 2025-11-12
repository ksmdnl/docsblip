devices=[2]

ckpt=logs/docsblip_stage1/runs/2025-11-11_19-30-14
model=src.models.docsblip_stage2.DocsBlip
task_name=docsblip_stage2
tags=[docvqa,docsblip,stage2]

ref_lr=1e-4
start_lr=1e-6
final_lr=1e-5
ref_wd=0.05
warmup_steps=2000

python src/train.py \
    +matmul_precision=medium \
    trainer.devices=${devices}\
    trainer.max_epochs=10 \
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
    +model.ckpt_path=${ckpt} \
    experiment=docsblip \
    data=docvqa \
    +data.num_workers=8 \
    +data.batch_size=32 \
    callbacks.rich_progress_bar=null \
    task_name=${task_name} \
    +logger.wandb.name=${task_name} \
    logger.wandb.tags=${tags} \
    # logger.wandb=null
    # logger.wandb=null \
    # +data.subset_size=200 \
