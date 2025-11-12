devices=[2]
tags=[docvqa,docsblip,naive]

task_name=docsblip_naive
task_name=debug
tags=[docvqa,docsblip]

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
    +trainer.num_sanity_val_steps=1 \
    trainer=gpu \
    model=docsblip \
    model.ref_lr=${ref_lr} \
    model.start_lr=${start_lr} \
    model.final_lr=${final_lr} \
    model.ref_wd=${ref_wd} \
    model.warmup_steps=${warmup_steps} \
    experiment=docsblip \
    data=docvqa \
    +data.num_workers=8 \
    +data.batch_size=32 \
    callbacks.rich_progress_bar=null \
    task_name=${task_name} \
    +logger.wandb.name=${task_name} \
    logger.wandb.tags=${tags} \
    logger.wandb.name=docsblip\
