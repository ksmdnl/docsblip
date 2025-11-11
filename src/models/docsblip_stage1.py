from typing import Any, Dict, Tuple
import contextlib

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from transformers import LiltModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from src.models.optim import ClipLoss
from src.utils.schedulers import WarmupCosineSchedule
from src.utils.caption_eval.custom_caption_eval import calculate_metrics
from src.models.blip_base import init_Qformer, init_tokenizer

import logging
logger = logging.getLogger(__name__)

from typing import List

class DocsBlip(LightningModule):
    def __init__(
        self,
        encoder_name="SCUT-DLVCLab/lilt-roberta-en-base",
        decoder_name="meta-llama/Llama-2-7b-chat-hf",
        max_len=32,
        start_lr=1e-8,
        ref_lr=1e-5,
        final_lr=0,
        ref_wd=0.04,
        final_wd=0.4,
        dropout=0.1,
        warmup_steps=1000,
        alpha=0.5,
        num_beams: int = 5,
        min_len: int = 10,
        repetition_penalty: float = 1.5,
        length_penalty: int = 1,
        top_p: float = 0.9,
        temperature: float = 1.0,
        num_query: int = 32,
        cross_att_freq: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = init_tokenizer()

        self.encoder = LiltModel.from_pretrained(encoder_name)
        for n, p in self.encoder.named_parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.Qformer, self.query_tokens = init_Qformer(
            num_query, self.encoder.config.hidden_size, cross_att_freq,
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.doc_proj = nn.Linear(self.Qformer.config.hidden_size, 256)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, 256)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_doc(self):
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
        loss = self(
            batch.input_ids,
            batch.input_attention_mask,
            batch.input_bbox,
            batch.questions,
            batch.targets,
        )
        for k, v in loss.items():
            self.log(k, v, prog_bar=True, logger=True)
        self.log("lr", self.lr_schedulers().get_last_lr()[0])
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        input_attention_mask: torch.Tensor,
        input_bbox: torch.Tensor,
        queries: List,
        targets: List,
    ) -> torch.Tensor:
        bs = input_ids.shape[0]
        device = input_ids.device

        # with self.maybe_autocast():
        doc_emb = self.encoder(input_ids=input_ids, bbox=input_bbox, attention_mask=input_attention_mask).last_hidden_state
        doc_emb = F.normalize(doc_emb, dim=-1)
        
        query_tokens = self.query_tokens.expand(doc_emb.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=doc_emb,
            encoder_attention_mask=input_attention_mask,
            use_cache=True,
            return_dict=True,
        )
        doc_feats = F.normalize(
            self.doc_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            targets,
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_len,
            return_tensors="pt",
        ).to(device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        sim_q2t = torch.matmul(
            doc_feats.unsqueeze(1), text_feat.unsqueeze(-1)
        ).squeeze()

        # image-text similarity: aggregate across all query tokens
        sim_d2t, _ = sim_q2t.max(-1)
        sim_d2t = sim_d2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), doc_feats.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2d, _ = sim_t2q.max(-1)
        sim_t2d = sim_t2d / self.temp  # [batch_size, batch_size*num_gpu]

        targets_ = torch.linspace(0, bs - 1, bs, dtype=int).to(device)

        loss_dtc = (
            F.cross_entropy(sim_d2t, targets_, label_smoothing=0.1)
            + F.cross_entropy(sim_t2d, targets_, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###
        with torch.no_grad():
            sim_t2d[:, :bs].fill_diagonal_(-10000)
            sim_d2t[:, :bs].fill_diagonal_(-10000)            
                
            weights_t2d = F.softmax(sim_t2d, dim=1)
            weights_d2t = F.softmax(sim_d2t, dim=1)

        # select a negative image for each text
        doc_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2d[b], 1).item()
            doc_embeds_neg.append(doc_emb[neg_idx])
        doc_embeds_neg = torch.stack(doc_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_d2t[b], 1).item()
            text_ids_neg.append(text_tokens.input_ids[neg_idx])
            text_atts_neg.append(text_tokens.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        doc_embeds_all = torch.cat(
            [doc_emb, doc_embeds_neg, doc_emb], dim=0
        )  # pos, neg, pos
        doc_atts_all = torch.ones(doc_embeds_all.size()[:-1], dtype=torch.long).to(device)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=doc_embeds_all,
            encoder_attention_mask=doc_atts_all,
            return_dict=True,
        )

        dl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        dl_output = self.itm_head(dl_embeddings)
        logits = dl_output.mean(dim=1)

        dtm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)
        loss_dtm = F.cross_entropy(logits, dtm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        return {
            "loss": loss_dtc + loss_dtm + loss_lm,
            "loss_dtc": loss_dtc,
            "loss_dtm": loss_dtm,
            "loss_lm": loss_lm,
        }

    def configure_optimizers(self):
        iterations_per_epoch = self.trainer.num_training_batches
        if iterations_per_epoch == float("inf"):
            iterations_per_epoch = len(self.trainer.datamodule.train_dataloader())

        num_epochs = self.trainer.max_epochs
        param_groups = self._get_param_groups()
        optimizer = AdamW(
            params=param_groups,
            lr=self.hparams.ref_lr,
            weight_decay=self.hparams.ref_wd,
        )
        lr_scheduler = WarmupCosineSchedule(
            optimizer,
            start_lr=self.hparams.start_lr,
            ref_lr=self.hparams.ref_lr,
            final_lr=self.hparams.final_lr,
            warmup_steps=self.hparams.warmup_steps,
            T_max=int(num_epochs * iterations_per_epoch),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(device_type=self.device.type, dtype=dtype)
        else:
            return contextlib.nullcontext()

    # def on_save_checkpoint(self, checkpoint):
    #     breakpoint()
    #     # module_names = list(checkpoint['state_dict'])
    #     # if 'decoder' in module_names:
    #     #     checkpoint['state_dict'].pop('lang_decoder')
    #     # if 'decoder' in module_names:
    #     #     checkpoint['state_dict'].pop('lang_decoder')
    #     checkpoint['state_dict'].pop('encoder')

    def _get_param_groups(self):
        return [
            {
                "params": (
                    p
                    for n, p in self.Qformer.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                )
            },
            {
                "params": (
                    p
                    for n, p in self.Qformer.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
            {
                "params": (
                    p
                    for n, p in self.doc_proj.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                )
            },
            {
                "params": (
                    p
                    for n, p in self.doc_proj.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
            {
                "params": (
                    p
                    for n, p in self.text_proj.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                )
            },
            {
                "params": (
                    p
                    for n, p in self.text_proj.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
            {
                "params": (
                    p
                    for n, p in self.itm_head.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                )
            },
            {
                "params": (
                    p
                    for n, p in self.itm_head.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
            {'params': self.logit_scale, 'WD_exclude': True, 'weight_decay': 0},
            {'params': self.query_tokens, 'WD_exclude': True, 'weight_decay': 0},
        ]

if __name__ == "__main__":
    _ = DocsBlip()