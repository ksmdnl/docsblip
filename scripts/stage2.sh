devices=[2]

ckpt=logs/docsblip_stage1/runs/2025-11-10_23-53-07
model=src.models.docsblip_stage2.DocsBlip
task_name=docsblip_stage2
tags=[docvqa,docsblip]

ref_lr=1e-5
start_lr=1e-6
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
    +model.ckpt_path=${ckpt} \
    experiment=docsblip \
    data=docvqa \
    +data.num_workers=8 \
    +data.batch_size=32 \
    callbacks.rich_progress_bar=null \
    +logger.wandb.name=${task_namek}\
    logger.wandb.tags=${tags}\
    task_name=${task_name}\
    # +data.subset_size=200 \
    # logger.wandb=null
