from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from transformers import AutoTokenizer, EncoderDecoderModel, AutoConfig, LiltModel, BartForConditionalGeneration

class LiLTAdapter(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim)
        # optional layernorm / dropout
        self.ln = torch.nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.ln(self.proj(x))

class LiLTEncoder(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.hidden_size = 768
        self.model = LiltModel.from_pretrained(pretrained_path)

    def forward(self, images, boxes, texts):
        # expected to return (batch, seq_len, hidden_size)
        outputs = self.model(images=images, boxes=boxes, texts=texts)
        return outputs.last_hidden_state

class DocsBlip(LightningModule):
    def __init__(
        self, encoder_name="SCUT-DLVCLab/lilt-roberta-en-base", decoder_name="facebook/bart-base"
    ) -> None:
        super().__init__()
        # self.doc_encoder = LiltModel.from_pretrained("niels/fundsr")
        self.encoder = LiltModel.from_pretrained(encoder_name)
        self.decoder = BartForConditionalGeneration.from_pretrained(decoder_name)
        self.adapter = LiLTAdapter(
            in_dim=self.encoder.hidden_size,
            out_dim=self.decoder.config.d_model
        )

        self.save_hyperparameters()

    def encode_doc(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        pass

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        pass

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        pass

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @torch.no_grad()
    def generate(self, images, boxes, texts, tokenizer, **gen_kwargs):
        enc_out = self.encoder(images, boxes, texts)
        enc_proj = self.adapter(enc_out)
        outputs = self.decoder.generate(
            encoder_outputs=(enc_proj,),
            **gen_kwargs
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded


if __name__ == "__main__":
    _ = DocsBlip()
