import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np

# debugger
from PIL import ImageDraw
import matplotlib.pyplot as plt

from transformers import LiltForTokenClassification, LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor

from net.model import LiLT
from utils.util import convert_pdf_to_images
from inference.parsing import *

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def unnormalize_box(bbox: List, width: int, height: int) -> List:
  """Retrieve the original size from predictions."""
  return [
    width * (bbox[0] / 1000),
    height * (bbox[1] / 1000),
    width * (bbox[2] / 1000),
    height * (bbox[3] / 1000),
  ]

def parse_ner_predictions(
    id2label_ner: Dict[int, str],
    offset_mapping: List,
    predictions: List,
    token_boxes: List,
    input_ids: List,
    tokenizer: LayoutLMv3TokenizerFast,
    dim: Tuple[int],
    img = None,
) -> Dict[Any, Any]:
  """Parse the predictions and return key-value pairs."""
  width, height = dim
  is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
  true_predictions = [id2label_ner[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
  true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

  input_words = [tokenizer.decode(i) for i in input_ids]

  # Decode true words from subwords
  true_words = decode_words(input_words, is_subword)

  true_predictions = true_predictions[1:-1]
  true_boxes = true_boxes[1:-1]
  true_words = true_words[1:-1]

  # Remove No Entity predictions
  preds, l_words, _ = remove_no_entity(true_predictions, true_words, true_boxes)

  data = {}
  for id, i in enumerate(preds):
    if i not in data.keys():
      data[i] = {"val": [l_words[id]]}
    else:
      data[i]["val"].append(l_words[id])

  return data

def parse_table_predictions(
    id2label_ner: Dict[int, str],
    offset_mapping: List,
    predictions: List,
    token_boxes: List,
    input_ids: List,
    tokenizer: LayoutLMv3TokenizerFast,
    dim: Tuple[int],
    row_boxes: List = None,
    img = None,
) -> Dict[Any, Any]:
  """Parse the predictions and return key-value pairs."""
  width, height = dim
  is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
  true_predictions = [id2label_ner[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
  true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
  input_words = [tokenizer.decode(i) for i in input_ids]

  # Decode true words from subwords
  true_words = decode_words(input_words, is_subword)

  true_predictions = true_predictions[1:-1]
  true_boxes = true_boxes[1:-1]
  true_words = true_words[1:-1]

  # Remove No Entity predictions
  preds, l_words, boxes = remove_no_entity(true_predictions, true_words, true_boxes)

  # Remove invalid predictions
  # Invalid means here degenerated boxes
  valid_preds, valid_words, valid_boxes = remove_invalid_boxes(preds, l_words, boxes)

  # Assign position of each word according to predicted row boxes
  l_words_matched = assign_pos(valid_boxes, valid_words, valid_preds, row_boxes)

  
  # image_viz = img.copy()
  # draw = ImageDraw.Draw(image_viz)
  # for box in row_boxes: draw.rectangle(box, outline='red', width=5)
  # for box in valid_boxes: draw.rectangle(box, outline='blue', width=5)
  # plt.figure(figsize=(10, 20));plt.imshow(image_viz); plt.show()
  # breakpoint()


  print("length of l_words_matched:", len(l_words_matched), " length of valid_preds:", len(valid_preds))
  # Generate the key-value pairs
  data = {}
  for id, (word, pos, cat) in enumerate(l_words_matched):
    if cat not in data.keys():
      data[cat] = {'val': [word], 'pos': [pos]}
    else:
      data[cat]['val'].append(word)
      data[cat]['pos'].append(pos)

  # Grouped words of the same position
  # TODO: Refactor this
  max_pos = -1
  for _, rows in data.items():
    grouped = defaultdict(list)
    for val, pos in zip(rows["val"], rows["pos"]):
      grouped[pos].append(val)
    merged_result = {pos: ' '.join(values) for pos, values in grouped.items()}
    new_pos = list(merged_result.keys())
    new_val = list(merged_result.values())
    rows["pos"] = new_pos
    rows["val"] = new_val
    max_pos = max(max_pos, max(new_pos))

  # Reconstruct the table structure
  max_pos += 1
  for col, rows in data.items():
    reconstructed_pos = list(range(max_pos))
    reconstructed_val = [""] * max_pos
    for val, pos in zip(rows["val"], rows["pos"]):
      reconstructed_val[pos] = val
    rows["pos"] = reconstructed_pos
    rows["val"] = reconstructed_val
  return data

def load_model(
    ckpt_path: str,
    label_values: Dict[Any, Any],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    tessconfig: Dict,
) -> Dict[Any, Any]:
  model_id= "nielsr/lilt-xlm-roberta-base"
  ner = LiltForTokenClassification.from_pretrained(model_id, label2id=label2id, id2label=id2label, num_labels=len(label_values))
  model = LiLT.load_from_checkpoint(checkpoint_path=ckpt_path, net=ner, train_dataloader=None, processor=None)

  # forward pass
  tokenizer_id = "xlm-roberta-base"
  tokenizer = LayoutLMv3TokenizerFast.from_pretrained(tokenizer_id)
  feature_extractor = LayoutLMv3FeatureExtractor(
    apply_ocr=True, ocr_lang=tessconfig["lang"], tesseract_config=tessconfig["config"])
  processor = LayoutLMv3Processor(feature_extractor, tokenizer=tokenizer)

  return {
    "tokenizer": tokenizer,
    "processor": processor,
    "classifier": model.model,
  }

def main() -> None:
  path = "../data/benchmarking/training/docs/IMG_3997_processed_2.pdf"
  image = convert_pdf_to_images(path)[0]

  label_values, id2label_ner, label2id_ner = get_ner_labels()
  model = load_model(label_values, label2id_ner, id2label_ner)

  encoding = model["processor"](image, return_offsets_mapping=True, return_tensors="pt")
  offset_mapping = encoding.pop('offset_mapping')
  encoding.pop('pixel_values')

  for k, v in encoding.items():
    encoding[k] = v.to('cuda')

  outputs = model["classifier"](**encoding)
  predictions = outputs.logits.argmax(-1).squeeze().tolist()
  token_boxes = encoding.bbox.squeeze().tolist()
  input_ids = encoding.squeeze().tolist()

  img_size = tuple(image.size)

  data = parse_ner_predictions(
    id2label_ner, offset_mapping, predictions, token_boxes, input_ids, model["tokenizer"], img_size)

  for k, v in data.items():
    print(k, v)

if __name__ == "__main__":
  main()