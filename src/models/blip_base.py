import torch
import torch.nn as nn
from transformers import BertTokenizer

from lightning import LightningDataModule

from src.models.qformer import BertConfig, BertLMHeadModel

# class BlipBase(LightningDataModule):
def init_tokenizer(truncation_side="right"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    return tokenizer

def init_Qformer(num_query_token, vision_width, cross_attention_freq=2):
    encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    encoder_config.encoder_width = vision_width
    # insert cross-attention layer every other block
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    Qformer = BertLMHeadModel.from_pretrained(
        "bert-base-uncased", config=encoder_config
    )
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens