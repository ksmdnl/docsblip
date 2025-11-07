tags=[docvqa,docsblip]
python src/train.py \
    +matmul_precision=medium \
    trainer.max_epochs=5 \
    +trainer.precision=16 \
    +trainer.num_sanity_val_steps=1 \
    trainer=gpu \
    model=docsblip \
    experiment=docsblip \
    data=docvqa \
    +data.subset_size=200 \
    +data.num_workers=2 \
    callbacks.rich_progress_bar=null \
    logger.wandb=null
    # +logger.wandb.name=docsblip\
    # logger.wandb.tags=${tags}\
