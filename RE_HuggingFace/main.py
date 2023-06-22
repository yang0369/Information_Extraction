from collections import defaultdict
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
from PIL import Image, JpegImagePlugin
from torch.utils.data import DataLoader

JpegImagePlugin._getmp = lambda: None
import matplotlib
import numpy as np
import torch
# enable only if using DGX machine to plot visuals
import tornado
from datasets import load_dataset
from pytz import timezone

from transformers import (AutoModelForTokenClassification, AutoProcessor,
                          AutoTokenizer, LayoutLMForRelationExtraction,
                          LayoutLMv2FeatureExtractor,
                          LayoutLMv2ForRelationExtraction,
                          LayoutLMv2ForTokenClassification,
                          PreTrainedTokenizerBase, TrainingArguments)
from transformers.file_utils import PaddingStrategy
from unilm.layoutlmft.layoutlmft.evaluation import re_score
from unilm.layoutlmft.layoutlmft.trainers import XfunReTrainer

matplotlib.use('WebAgg')

torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).parents[1]

TZ = timezone('Asia/Singapore')
CURRENT = datetime.now(tz=TZ)
TIME = CURRENT.strftime("%Y_%m_%d_%H_%M")
MODDEL_DIR = ROOT / "RE_HuggingFace" / f"model/checkpoint_{TIME}.pt"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger_path = ROOT / "RE_HuggingFace" / "artifacts" / f"experiment_{TIME}.log"
fileHandler = logging.FileHandler(f"{logger_path.as_posix()}")
logFormatter = logging.Formatter("[%(levelname)s] - %(message)s")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)


# @dataclass
# class DataCollatorForKeyValueExtraction:
#     """
#     Data collator that will dynamically pad the inputs received, as well as the labels.

#     Args:
#         tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
#             The tokenizer used for encoding the data.
#         padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:
#             * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
#               sequence if provided).
#             * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
#               maximum acceptable input length for the model if that argument is not provided.
#             * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
#               different lengths).
#         max_length (:obj:`int`, `optional`):
#             Maximum length of the returned list and optionally padding length (see above).
#         pad_to_multiple_of (:obj:`int`, `optional`):
#             If set will pad the sequence to a multiple of the provided value.
#             This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
#             7.5 (Volta).
#         label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
#             The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
#     """
#     feature_extractor: LayoutLMv2FeatureExtractor
#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     label_pad_token_id: int = -100

#     def __call__(self, features):
#         # prepare image input
#         image = self.feature_extractor([feature["original_image"] for feature in features], return_tensors="pt").pixel_values

#         # prepare text input
#         entities = []
#         relations = []
#         for feature in features:
#             del feature["image"]
#             del feature["id"]
#             del feature["labels"]
#             del feature["original_image"]
#             entities.append(feature["entities"])
#             del feature["entities"]
#             relations.append(feature["relations"])
#             del feature["relations"]

#         batch = self.tokenizer.pad(
#             features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt"
#         )

#         batch["image"] = image
#         batch["entities"] = entities
#         batch["relations"] = relations

#         return batch


# def unnormalize_box(bbox, width, height):
#     return [
#          width * (bbox[0] / 1000),
#          height * (bbox[1] / 1000),
#          width * (bbox[2] / 1000),
#          height * (bbox[3] / 1000),
#     ]


# def compute_metrics(p):
#     pred_relations, gt_relations = p
#     score = re_score(pred_relations, gt_relations, mode="boundaries")
#     return score


# dataset = load_dataset(path=(ROOT / "RE_HuggingFace" / "download.py").as_posix(), name="en")

# # model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutxlm-base")
# # tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")


# tokenizer = AutoTokenizer.from_pretrained(ROOT / "RE_HuggingFace/model/checkpoint_2023_06_06_14_17.pt/checkpoint-5000")
# # model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutlmv2-base-uncased")
# model = LayoutLMv2ForRelationExtraction.from_pretrained(ROOT / "RE_HuggingFace/model/checkpoint_2023_06_06_14_17.pt/checkpoint-5000")


# # model = LayoutLMForRelationExtraction.from_pretrained("microsoft/layoutlm-base-uncased")
# # tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")


# feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)

# data_collator = DataCollatorForKeyValueExtraction(
#     feature_extractor,
#     tokenizer,
#     pad_to_multiple_of=1,
#     padding="max_length",
#     max_length=512,
# )

# train_dataset = dataset['train']
# val_dataset = dataset['validation']

# dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=data_collator)

# # Define TrainingArguments
# # See thread for hyperparameters: https://github.com/microsoft/unilm/issues/586
# training_args = TrainingArguments(output_dir=MODDEL_DIR,
#                                   overwrite_output_dir=True,
#                                   remove_unused_columns=False,
#                                   # fp16=True, -> led to a loss of 0
#                                   # num_train_epochs=1,
#                                   max_steps=5000,
#                                   no_cuda=False,
#                                   per_device_train_batch_size=2,
#                                   per_device_eval_batch_size=1,
#                                   warmup_ratio=0.1,
#                                   learning_rate=1e-5,
#                                   push_to_hub=False,
#                                   )

# # Initialize our Trainer
# trainer = XfunReTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# logger.info("start training model")
# train_metrics = trainer.train()
# logger.info(f"training_metrics: {train_metrics}")

# logger.info("start evaluating performance")
# eval_metrics = trainer.evaluate()
# logger.info(f"evaluation metrics: {eval_metrics}")

#################################### Inference Pipeline ####################################
# test_image = train_dataset[48]['original_image']
test_image = Image.open(ROOT / 'dataset/test/1.jpg')
# plt.imshow(test_image)
# plt.show()

# load model + processor from the hub
processor = AutoProcessor.from_pretrained(ROOT / "SER_HuggingFace" / "model")
model = AutoModelForTokenClassification.from_pretrained(ROOT / "SER_HuggingFace" / "model")
# prepare inputs for the model
# we set `return_offsets_mapping=True` as we use the offsets to know which tokens are subwords and which aren't
inputs = processor(test_image, return_offsets_mapping=True, padding="max_length", max_length=512, truncation=True, return_tensors="pt")

original_text = processor.tokenizer.convert_tokens_to_string([processor.tokenizer.decode(i, skip_special_tokens=True) for i in inputs["input_ids"][0].tolist()])
# all_token_text = [processor.tokenizer.decode(i, skip_special_tokens=True) for i in inputs["input_ids"][0].tolist()]

inputs = inputs.to(DEVICE)
model.to(DEVICE)

# offset_mapping: indicates the start and end index of the actual subword w.r.t each token text, e.g. '##omi' with offset [2, 5] -> 'omi'
offset_mapping = inputs.pop("offset_mapping")

# word_ids: indicates if the subtokens belong to the same word.
word_ids = inputs.encodings[0].word_ids

token_ids = inputs.input_ids[0].tolist()

if_special = inputs.encodings[0].special_tokens_mask

# forward pass
with torch.no_grad():
  outputs = model(**inputs)

# take argmax on last dimension to get predicted class ID per token
predictions = outputs.logits.argmax(-1).squeeze().tolist()

# # check if it's subwords
# is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0

# merge subwords into word-level based on word_ids
word_pred = defaultdict(lambda: -1)
words = defaultdict(list)
for idx, tp in enumerate(zip(if_special, token_ids, predictions, word_ids)):
  if idx == 0 or bool(tp[0]):
    continue

  words[tp[-1]].append(idx)
  if word_pred[tp[-1]] == -1:
    word_pred[tp[-1]] = tp[2]

id2label = {"QUESTION": 0, "ANSWER": 1}

# finally, store recognized "question" and "answer" entities in a list
entities = []
current_entity = None
start = None
end = None
for idx, (id, pred) in enumerate(zip(words.values(), word_pred.values())):
  predicted_label = model.config.id2label[pred]

  if predicted_label.startswith("B") and current_entity is None:
    # means we're at the start of a new entity
    current_entity = predicted_label.replace("B-", "")
    start = min(id)
    print(f"--------------New entity: at index {start}", current_entity)

  if current_entity is not None and current_entity not in predicted_label:
    # means we're at the end of a new entity
    end = max(words[idx - 1])
    print("---------------End of new entity")
    entities.append((start, end, current_entity, id2label[current_entity]))
    current_entity = None

    if predicted_label.startswith("B") and current_entity is None:
      # means we're at the start of a new entity
      current_entity = predicted_label.replace("B-", "")
      start = min(id)
      print(f"--------------New entity: at index {start}", current_entity)

# #################################### Load our dataset ####################################
# step 2: run LayoutLMv2ForRelationExtraction
relation_extraction_model = LayoutLMv2ForRelationExtraction.from_pretrained(ROOT / "RE_HuggingFace/model/checkpoint_2023_06_22_11_48.pt/checkpoint-5000")
relation_extraction_model.to(DEVICE)

# with open(ROOT / "dataset" / "test.json", "rb") as f:
#   file = json.load(f)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
# entity_dict = file["entity_dict"]
# inputs_ = file["input"]

# for k, v in inputs_.items():
#   inputs_[k] = torch.tensor(inputs_[k])

# # inputs_["image"] = inputs_["image"].permute(0, 3, 1, 2)

# # use processor input temporarily
# inputs_["image"] = inputs.data["image"]
# inputs_["bbox"] = inputs.data["bbox"]

entity_dict = {'start': [entity[0] for entity in entities],
        'end': [entity[1] for entity in entities],
        'label': [entity[3] for entity in entities]}

with torch.no_grad():
  # inputs: {'input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'}
  outputs = relation_extraction_model(**inputs,
                                      entities=[entity_dict],
                                      relations=[{'start_index': [], 'end_index': [], 'head': [], 'tail': []}])

# show predicted key-values
for relation in outputs.pred_relations[0]:
  head_start, head_end = relation['head']
  tail_start, tail_end = relation['tail']
  print("Question:", processor.decode(inputs.input_ids[0][head_start:head_end]))
  print("Answer:", processor.decode(inputs.input_ids[0][tail_start:tail_end]))
  print("----------")

# for relation in outputs.pred_relations[0]:
#   head_start, head_end = relation['head']
#   tail_start, tail_end = relation['tail']
#   print("Question:", tokenizer.decode(inputs_["input_ids"][0][head_start:head_end]))
#   print("Answer:", tokenizer.decode(inputs_["input_ids"][0][tail_start:tail_end]))
#   print("----------")