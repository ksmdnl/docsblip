import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset, load_dataset

from transformers import AutoTokenizer, LayoutLMv3FeatureExtractor, LayoutLMv3Processor, LayoutLMv3TokenizerFast, LiltModel
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from lightning import LightningDataModule

from typing import Optional

import logging
logger = logging.getLogger(__name__)

def subfinder(words_list, answer_list):
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if words_list[i] == answer_list[0] and words_list[i : i + len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0
    
def clean_text(text):
    replace_chars = ',.;:()/-$%&*' 
    for j in replace_chars:
        if text is not None:
            text = text.replace(j,'')
    return text

def encode_dataset(examples, max_length=32, tokenizer=None, decoder_tokenizer=None):
    # questions = examples["question"]
    words = examples["words"]
    boxes = examples["bbox"]
    answers = examples["answer"]
    
    encoding = tokenizer(
        # questions,
        words,
        boxes=boxes,
        truncation="only_second",
        padding="max_length",
        # return_tensors='pt',
        # return_token_type_ids=True,
    )
    answer_tokens = decoder_tokenizer(
        answers,
        padding=True,
        truncation='longest_first',
        return_tensors='pt',
        max_length=max_length,
        padding_side='left',
    )
    encoding['answer_token_ids'] = answer_tokens.input_ids
    encoding['answer_attention_mask'] = answer_tokens.attention_mask
    return encoding

def normalize_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(x1*1000 // width, 1000))
    y1 = max(0, min(y1*1000 // height, 1000))
    x2 = max(0, min(x2*1000 // width, 1000))
    y2 = max(0, min(y2*1000 // height, 1000))
    return x1, y1, x2, y2

def generate_dataset(
    annotation_path: str,
    ocr_path: str,
    subset_size: int = None,
):
    """This module will combine question and answers from the data excluding image as LayoutLM is independent of Image, it
    requires only text and box embeddings
    
    """
    with open(annotation_path, 'r') as f:
        train_json = json.load(f)
    dataset_list = []
    
    if subset_size:
        train_json['data'] = train_json['data'][:subset_size]

    for data in tqdm(train_json['data']):
        ans_json_name = os.path.join(ocr_path, data['image'].split('/')[-1].split('.')[0]) + '.json'

        with open(ans_json_name, 'r') as f:
            ans_json = json.load(f)

        answer_dict = ans_json['recognitionResults']
        question = data['question']
        
        local_dict = {}
        local_dict['id'] = 'id_' + data['image'].split('/')[-1].split('.')[0]
        local_dict['question'] = question
        local_dict['answer'] = data['answers'][0]
        
        words = []
        boxes = []
        for obj in answer_dict:
            width, length = obj['width'], obj['height']
            lines = obj['lines']

            for line in lines:
                for word in line['words']:
                    words.append(word['text'].lower())    
                    x1, y1, x2, y2, x3, y3, x4, y4 = word['boundingBox']
                    new_x1 = min([x1, x2, x3, x4])
                    new_x2 = max([x1, x2, x3, x4])
                    new_y1 = min([y1, y2, y3, y4])
                    new_y2 = max([y1, y2, y3, y4])
                    
                    box_norm = normalize_box(new_x1, new_y1, new_x2, new_y2, width, length)
                    assert new_x2 >= new_x1
                    assert new_y2 >= new_y1
                    assert box_norm[2] >= box_norm[0]
                    assert box_norm[3] >= box_norm[1]

                    boxes.append(box_norm)
                    
        local_dict['words'] = words
        local_dict['bbox'] = boxes
        
        dataset_list.append(local_dict)
        
    hf_data = Dataset.from_list(dataset_list)
    return hf_data

from dataclasses import dataclass

from typing import List

@dataclass
class BatchContainer:
    input_ids: torch.Tensor
    input_attention_mask: torch.Tensor
    input_bbox: torch.Tensor
    questions: List
    targets: List
    words: List

class Collater(object):
    def __init__(self, encoder_tokenizer_path: str, max_len: int = 32):
        self.max_len = max_len
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_path)

    def __call__(self, batch):
        questions = []
        answers = []
        boxes = []
        words = []
        for b in batch:
            questions.append(b['question'])
            answers.append(b['answer'])
            boxes.append(b['bbox'])
            words.append(b['words'])

        try:
            encoding = self.encoder_tokenizer(
                words,
                boxes=boxes,
                truncation="longest_first",
                padding="max_length",
                return_tensors='pt',
                # return_token_type_ids=True,
                max_length=self.max_len,
            )
        except Exception as e:
            logger.error(e)
            breakpoint()
        input_ids = encoding['input_ids']
        input_attention_mask = encoding['attention_mask']
        input_bbox = encoding['bbox']
        return BatchContainer(
            input_ids=input_ids,
            input_attention_mask=input_attention_mask,
            input_bbox=input_bbox,
            targets=answers,
            questions=questions,
            words=words,
        )

class VQADatamodule(LightningDataModule):
    def __init__(
        self,
        train_annotation_path: str,
        val_annotation_path: str,
        ocr_path: str,
        subset_size: int = None,
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.collate_fn = Collater(
            encoder_tokenizer_path="SCUT-DLVCLab/lilt-roberta-en-base",
            max_len=32,
        )
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_set = generate_dataset(
                annotation_path=self.hparams.train_annotation_path,
                ocr_path=self.hparams.ocr_path,
                subset_size=self.hparams.subset_size,
            )
            self.val_set = generate_dataset(
                annotation_path=self.hparams.val_annotation_path,
                ocr_path=self.hparams.ocr_path,
                subset_size=self.hparams.subset_size,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

if __name__ == "__main__":
    dataset = generate_dataset(
        annotation_path="data/spdocvqa_qas/train_v1.0_withQT.json",
        ocr_path="data/ocr",
        subset_size=100,
    )
    # dataset = generate_dataset(
    #     annotation_path="data/spdocvqa_qas/train_v1.0_withQT.json",
    #     ocr_path="data/ocr",
    #     subset_size=100,
    # )
    # tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
    # decoder_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # fn_kwargs = {
    #     "tokenizer": tokenizer,
    #     "decoder_tokenizer": decoder_tokenizer,
    #     "max_length": 512,
    # }
    # encoded_train_dataset = dataset.map(
    #     encode_dataset, fn_kwargs=fn_kwargs, batched=True, batch_size=2, remove_columns=dataset.column_names
    # )
    collater = Collater(
        "SCUT-DLVCLab/lilt-roberta-en-base",
        "facebook/bart-base",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        # collate_fn=collate_fn,
        collate_fn=collater,
    )
    sample = next(iter(dataloader))
    breakpoint()