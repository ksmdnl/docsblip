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


from typing import List

def generate_closed_instr(instr):
    return "[INST] " + instr + " [/INST]"


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

class QFormerAdapter(nn.Module):
    def __init__(
        self, 
        num_queries=32, 
        d_img=768,
        d_llm=4096,
        num_layers=4, 
        num_heads=8, 
        ff_dim=768,
    ):
        super().__init__()

        self.query_embeds = nn.Parameter(
            torch.zeros(num_queries, d_img)
        )
        self.query_embeds.data.normal_(mean=0.0, std=0.02)

        self.llm_proj = nn.Linear(d_img, d_llm)

        encoder_layer = nn.TransformerEncoderLayer(
            # d_model=d_llm,
            d_model=d_img,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, img_emb):
        B = img_emb.size(0)

        queries = self.query_embeds.unsqueeze(0).expand(B, -1, -1)

        x = torch.cat([queries, img_emb], dim=1)

        out = self.transformer(x)
        out = self.llm_proj(out)

        return out[:, :queries.size(1), :]


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
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = LiltModel.from_pretrained(encoder_name)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            dtype=torch.float16,
            # load_in_8bit=True,
        )
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name, truncation_side='left', use_fast=False)
        for n, p in self.encoder.named_parameters():
            p.requires_grad = False
        self.encoder.eval()
        for n, p in self.decoder.named_parameters():
            p.requires_grad = False
        self.decoder.eval()

        self.decoder_tokenizer.padding_side = "left"
        # self.decoder_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.decoder_tokenizer.pad_token_id = self.decoder_tokenizer.eos_token_id

        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        self.terminators = [
            self.decoder_tokenizer.eos_token_id,
            self.decoder_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.adapter = QFormerAdapter(
            d_img=self.encoder.config.hidden_size,
            d_llm=self.decoder.config.hidden_size,
            num_layers=1,
            num_heads=1,
        )
        self.clip_loss = ClipLoss()
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
        breakpoint()
        self.decoder_tokenizer.padding_side = "right"
        bs = input_ids.shape[0]
        with self.maybe_autocast():
            doc_emb = self.encoder(input_ids=input_ids, bbox=input_bbox, attention_mask=input_attention_mask).last_hidden_state
            doc_emb = F.normalize(doc_emb, dim=-1)
        
        doc_emb = self.adapter(doc_emb)

        target_tokens = self.decoder_tokenizer(
            targets,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.hparams.max_len,
        ).to(input_ids.device)

        with self.maybe_autocast():
            outputs = self.decoder.model(input_ids=target_tokens.input_ids,attention_mask=target_tokens.attention_mask,output_hidden_states=True,return_dict=True)
        ans_emb = outputs.last_hidden_state[:, 0, :]

        pooled_doc_emb = doc_emb.mean(dim=1)
        proj_emb = F.normalize(pooled_doc_emb, dim=-1)
        ans_emb = F.normalize(ans_emb, dim=-1)
        contrastive_loss = self.clip_loss(proj_emb, ans_emb, self.logit_scale.exp())

        self.decoder_tokenizer.padding_side = "right"
        self.decoder_tokenizer.truncation_side = "left"
        text_input_tokens = self.decoder_tokenizer(
            [generate_closed_instr(q) for q in queries],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.hparams.max_len,
        ).to(input_ids.device)

        self.decoder_tokenizer.truncation_side = "right"
        text_output_tokens = self.decoder_tokenizer(
            [t + self.decoder_tokenizer.eos_token for t in targets],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.hparams.max_len,
        ).to(input_ids.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # Do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.decoder_tokenizer.pad_token_id, -100
        )

        # Do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(doc_emb.shape[:-1], dtype=torch.long).to(input_ids.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        with self.maybe_autocast():
            inputs_embeds = self.decoder.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = torch.cat([doc_emb, inputs_embeds], dim=1)
            decoder_att = torch.ones(doc_emb.size()[:-1], dtype=torch.long).to(doc_emb.device)
            attention_mask = torch.cat([decoder_att, llm_tokens['attention_mask']], dim=1)
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            
        lm_loss = outputs.loss
        loss = self.hparams.alpha * contrastive_loss + (1 - self.hparams.alpha) * lm_loss
        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "lm_loss": lm_loss,
        }

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:], # exclude the BOS token
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:], # exclude the BOS token
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def _get_param_groups(self):
        return [
            {
                "params": (
                    p
                    for n, p in self.adapter.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                )
            },
            {
                "params": (
                    p
                    for n, p in self.adapter.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
            {'params': self.logit_scale, 'WD_exclude': True, 'weight_decay': 0},
        ]

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
    def on_validation_start(self):
        self.answer_results = {'annotations': []}
        self.answer_gts = {'annotations': []}
        self.act_bs = self.trainer.datamodule.hparams.batch_size

    def validation_step(self, batch, batch_idx):
        bs = batch.input_ids.shape[0]
        answers = self.generate(
            batch,
            use_nucleus_sampling=False,
            num_beams=self.hparams.num_beams,
            max_length=self.hparams.max_len,
            min_length=self.hparams.min_len,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            top_p=self.hparams.top_p,
            temperature=self.hparams.temperature,
        )
        """Infer the image ids sequentially."""
        doc_ids = list(range(batch_idx * self.act_bs, batch_idx * self.act_bs + bs))
        for ans_res, ans_gt, img_id in zip(answers, batch.targets, doc_ids):
            self.caption_results["annotations"].append({"caption": ans_res, "image_id": img_id})
            self.caption_gts["annotations"].append({"caption": ans_gt, "image_id": img_id})

    def on_validation_epoch_end(self):
        sample_ids = range(len(self.answer_gts["annotations"]))
        metrics = calculate_metrics(sample_ids, self.answer_gts, self.answer_results)
        self.log_dict(metrics, sync_dist=True)
        self.answer_gts["annotations"].clear()
        self.answer_results["annotations"].clear()

    @torch.no_grad()
    def generate(self,
        batch,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=512,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        temperature=1,
    ):
        device = batch.input_ids.device
        self.decoder_tokenizer.padding_side = "left"
        doc_emb = self.encoder(
            batch.input_ids, bbox=batch.input_bbox, attention_mask=batch.input_attention_mask,
        ).last_hidden_state

        doc_emb = F.normalize(doc_emb, dim=-1)

        doc_emb = self.adapter(doc_emb)
        atts_opt = torch.ones(doc_emb.size()[:-1], dtype=torch.long).to(device)

        # Tokenize prompt
        inputs = self.decoder_tokenizer(
            [generate_closed_instr(q) for q in batch.questions],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=32,
        ).to(device)

        txt_embeds = self.decoder.get_input_embeddings()(inputs.input_ids)
        inputs_embeds = torch.cat([doc_emb, txt_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, inputs.attention_mask], dim=1)

        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            # max_length=max_length,
            max_new_tokens=max_length,
            min_length=min_length,
            pad_token_id=self.decoder_tokenizer.eos_token_id, 
            eos_token_id=self.terminators,
            # eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=1,
        )

        decoded = self.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(device_type=self.device.type, dtype=dtype)
        else:
            return contextlib.nullcontext()

if __name__ == "__main__":
    _ = DocsBlip()