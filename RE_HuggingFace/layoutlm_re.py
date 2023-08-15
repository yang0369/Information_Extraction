import copy
import enum
from functools import reduce
import json
import logging
from math import sqrt
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image, JpegImagePlugin
from torch.utils.data import DataLoader
from transformers import LayoutLMTokenizer
import pandas as pd
import wandb
JpegImagePlugin._getmp = lambda: None
import matplotlib
import tornado

matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
# enable only if using DGX machine to plot visuals
from datasets import load_dataset


from pytz import timezone
from transformers import (AutoModelForTokenClassification, AutoProcessor,
                          LayoutLMForRelationExtraction,
                          PreTrainedTokenizerBase,
                          TrainingArguments)
from transformers.file_utils import PaddingStrategy

from unilm.layoutlmft.layoutlmft.evaluation import re_score
from unilm.layoutlmft.layoutlmft.trainers import XfunReTrainer

torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).parents[1]

TZ = timezone('Asia/Singapore')
CURRENT = datetime.now(tz=TZ)
TIME = CURRENT.strftime("%Y_%m_%d_%H_%M")
MODDEL_DIR = ROOT / "RE_HuggingFace" / f"model/checkpoint_{TIME}.pt"
DEVICE = "cuda"
# DEVICE = "cpu"


@enum.unique
class Task(enum.Enum):
    FINETUNING = 0
    INFERENCE = 1
    XTRACT_INFER = 2


task = Task.XTRACT_INFER


@dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """
    tokenizer: PreTrainedTokenizerBase()
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # prepare text input
        entities = []
        relations = []
        for feature in features:
            del feature["image"]
            del feature["id"]
            del feature["labels"]
            del feature["original_image"]
            entities.append(feature["entities"])
            del feature["entities"]
            relations.append(feature["relations"])
            del feature["relations"]

        # set a break point here and check out the data schema based on Appendix
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # batch["image"] = image
        batch["entities"] = entities
        batch["relations"] = relations

        return batch


def relate_to_entity(to_merge: set, list_of_entities: list) -> list:
    """
    replace id with actual entity
    Args:
        to_merge: a set of all merging ids
        list_of_entities: list of entities
    Returns: list of entities to be merged, with no order
    """
    out = list()
    for e in list_of_entities:
        if e.get("id", "") in to_merge:
            out.append(e)
    return out


def merge_two_entities(e_0: dict, e_1: dict) -> dict:
    """
    merge two entities
    Args:
        e_0: entity
        e_1: entity
    Returns: the combined entity
    """
    if "bbox" not in e_0.keys():
        return e_1
    if "bbox" not in e_1.keys():
        return e_0

    # decide the base entity
    # compare the y0, smaller y0 -> base
    if e_0["bbox"][1] == e_1["bbox"][1]:
        if e_0["bbox"][0] <= e_1["bbox"][0]:
            base_entity = e_0
            adding_entity = e_1
        else:
            base_entity = e_1
            adding_entity = e_0
    elif e_0["bbox"][1] < e_1["bbox"][1]:
        base_entity = e_0
        adding_entity = e_1
    else:
        base_entity = e_1
        adding_entity = e_0

    base_entity = merge_text(base_entity, adding_entity)
    base_entity = merge_bbox(base_entity, adding_entity)
    return base_entity


def merge_text(base_entity: dict, adding_entity: dict) -> dict:
    """
    merge text from two entities
    Args:
        base_entity: the entity that text starts
        adding_entity: the entity to be added to base
    Returns: an entity with combined text
    """
    base_entity["text"] = base_entity.get("text", "") + " " + adding_entity.get("text", "")
    return base_entity


def merge_bbox(base_entity: dict, adding_entity: dict) -> dict:
    """
    merge bbox from two entities
    Args:
        base_entity: the entity that bbox starts
        adding_entity: the entity to be added to base
    Returns: an entity with combined bbox
    """
    base_entity["bbox"] = [
        min(base_entity["bbox"][0], adding_entity["bbox"][0]),
        min(base_entity["bbox"][1], adding_entity["bbox"][1]),
        max(base_entity["bbox"][2], adding_entity["bbox"][2]),
        max(base_entity["bbox"][3], adding_entity["bbox"][3])
    ]
    return base_entity


def unnormalize_box(bbox, width, height):
    return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
    ]


def compute_metrics(p):
    pred_relations, gt_relations = p
    score = re_score(pred_relations, gt_relations, mode="boundaries")
    return score


def post_process_entities(entity_list: list, threshold: int = 2) -> tuple:
    """
    perform some cleaning and format changing task on entity_list
    Args:
        entity_list: all the entities after pairing step
        threshold: the largest vertical difference between two merging entities
        if diff > threshold, then no merge for these two entities.
    Returns:
    """
    # relation diagram to illustrate multi-key and multi-val linking
    # Given key_0 and key_1 form a key, and its corresponding val is formed by val_0, val_1 and val_2,
    # their linking relation could be as below:
    #      / val_0        / val_0
    # key_0- val_1   key_1- val_1    val_0 - key_1, val_1 - key_1, val_2 - key_1
    #      \ val_2        \ val_2
    # caveat: for the linking in value entity, it only shows key_1 instead of key_1 and key_2

    # remove entities without any linking, and change key "box" to "bbox"
    unpaired = list()
    en_list_ori = copy.deepcopy(entity_list)
    for idx, entity in enumerate(en_list_ori):
        if "id" not in entity.keys():
            unpaired.append(idx)
            continue

        # change id type to str
        entity["id"] = str(entity["id"])

        # remove invalid entity
        if "linking" not in entity.keys() or len(entity["linking"]) == 0:
            unpaired.append(idx)
            continue

    entity_list = [i for idx, i in enumerate(en_list_ori) if idx not in unpaired]
    if len(unpaired) != 0:
        unpaired = [i for idx, i in enumerate(en_list_ori) if idx in unpaired]

    # all entities in the list should have "id" and "linking" keys afterwards
    merge_list = list()
    kv_indexes = [e["id"] for e in entity_list]

    def append_merge_set(merge_list: list,
                         merge_set: set) -> list:
        """
        append merge_set to merge_list
        Args:
            merge_list: merge_list
            merge_set: merge_set
        Returns: list
        """
        if len(merge_list) == 0:
            merge_list.append(merge_set)
            return merge_list

        if merge_set in merge_list:
            return merge_list

        # there is overlaps between existing and merge_set
        for i in merge_list:
            if len(i.union(merge_set)) < len(i) + len(merge_set):
                i.update(merge_set)
                return merge_list

        merge_list.append(merge_set)
        return merge_list

    def get_counterpart(link: List, entity: Dict) -> str:
        # get the counterpart from a link
        if len(link) != 2:
            raise ValueError("link is not between 2 entities")

        id = entity["id"]
        for i in link:
            if str(i) != id:
                return str(i)

    # merge answers linked to the same question
    for entity in entity_list:
        # find the question entity which links multiple answers
        if entity.get("label", "") == "question" and len(entity["linking"]) > 1:
            merge_set = set()
            for link in entity["linking"]:
                cp = get_counterpart(link, entity)
                if cp in kv_indexes:
                    merge_set.add(cp)

            if len(merge_set) >= 2:
                merge_list = append_merge_set(merge_list, merge_set)

    # merge questions with same linkings
    # for temp_dict, key: sorted list of counterparts; value: question id
    temp_dict = defaultdict(lambda: set())
    for idx, entity in enumerate(entity_list):
        if entity.get("label", "") == "question":
            cps = sorted([get_counterpart(i, entity) for i in entity["linking"]])
            temp_dict[tuple(cps)].add(entity["id"])

    # add ids of merging keys to merge_list
    for q_ids in temp_dict.values():
        if len(q_ids) > 1:
            merge_list = append_merge_set(merge_list, q_ids)

    if len(merge_list) > 0:
        for merge_set in merge_list:
            en_list = relate_to_entity(to_merge=merge_set, list_of_entities=entity_list)

            # filter the big gap in merge list
            en_list = sorted(en_list, key=lambda x: (x["bbox"][1], x["bbox"][0]))
            for idx in range(len(en_list) - 1):
                y_diff = en_list[idx + 1]["bbox"][1] - en_list[idx]["bbox"][3]
                char_height = abs(en_list[idx]["bbox"][3] - en_list[idx]["bbox"][1])
                if y_diff > threshold * char_height:
                    # stop merging at idx, **Noted that all entities are sorted by y0
                    en_list = en_list[:(idx + 1)]
                    break
            base = reduce(merge_two_entities, en_list)
            # add base and drop entities in merge_set
            entity_list = [e for e in entity_list if e.get('id', -1) not in merge_set]
            entity_list.append(base)

    # update valid ids
    kv_indexes = [e["id"] for e in entity_list]

    # change linking from list to single index
    for entity in entity_list:
        if len(entity["linking"]) == 1:
            entity["linking"] = get_counterpart(entity["linking"][0], entity)
        else:
            for link in entity["linking"]:
                index = get_counterpart(link, entity)
                if index in kv_indexes:
                    entity["linking"] = index
                    break

    keyvalue = [e for e in entity_list if isinstance(e["linking"], str)]
    return keyvalue, unpaired


if task.name == "FINETUNING":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    logger_path = ROOT / "RE_HuggingFace" / "artifacts" / f"experiment_{TIME}.log"
    fileHandler = logging.FileHandler(f"{logger_path.as_posix()}")
    logFormatter = logging.Formatter("[%(levelname)s] - %(message)s")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="RE",

        # track hyperparameters and run metadata
        config={
        "dataset": "RE_FUNSD",
        }
    )

    dataset = load_dataset(path=(ROOT / "RE_HuggingFace" / "download.py").as_posix(), name="en")

    # # check if any bbox value > 1000, use it only for debugging >1000 error
    # for i in range(len(dataset["train"]["bbox"])):
    #   print(max(torch.tensor(dataset["train"]["bbox"][i])[:, 1]))

    model_card = "/home/kewen_yang/Information_Extraction/RE_HuggingFace/model/checkpoint_2023_08_01_15_26.pt"
    # model_card = "microsoft/layoutlm-base-uncased"

    model = LayoutLMForRelationExtraction.from_pretrained(model_card)
    tokenizer = LayoutLMTokenizer.from_pretrained(model_card)

    logger.info(f"finetuning model on top of {model_card}")
    logger.info(f"finetuning with dataset - RE_Finetune_1")
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=1,
        padding="max_length",
        max_length=512,
    )

    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    # Define TrainingArguments
    # See thread for hyperparameters: https://github.com/microsoft/unilm/issues/586
    training_args = TrainingArguments(
        output_dir=MODDEL_DIR,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        # fp16=True, -> led to a loss of 0

        max_steps=20000,
        # max_steps = 10,
        evaluation_strategy="steps",

        # num_train_epochs=1,
        # evaluation_strategy="epoch",
        save_strategy="no",
        no_cuda=(DEVICE == "cpu"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        warmup_ratio=0.1,
        learning_rate=1e-5,
        push_to_hub=False,
        report_to="wandb"
        )

    # Initialize our Trainer
    trainer = XfunReTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logger.info("start training model")
    train_metrics = trainer.train(resume_from_checkpoint=False)
    logger.info(f"training_metrics: {train_metrics}")

    logger.info("start evaluating performance")
    eval_metrics = trainer.evaluate()
    logger.info(f"evaluation metrics: {eval_metrics}")
    trainer.save_model(MODDEL_DIR)

    learning_curve = pd.DataFrame(trainer.state.log_history)
    logger.info('\n\t' + learning_curve.to_string().replace('\n', '\n\t'))


elif task.name == "INFERENCE":
  """do inference by huggingface pipeline
  """

  # test_image = train_dataset[48]['original_image']
  test_image = Image.open(ROOT / 'dataset/test/AUTOVACSTORE-1-2-Bing-image_010.jpg')
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

  id2label = {"QUESTION": 1, "ANSWER": 2}

  # finally, store recognized "question" and "answer" entities in a list
  entities = []
  current_entity = None
  start = None
  end = None

  for idx, (id, pred) in enumerate(zip(words.values(), word_pred.values())):
    predicted_label = model.config.id2label[pred]
    if predicted_label == "O":
        continue

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

  # step 2: run LayoutLMForRelationExtraction
  entity_dict = {'start': [entity[0] for entity in entities],
          'end': [entity[1] for entity in entities],
          'label': [entity[3] for entity in entities]}

  relation_extraction_model = LayoutLMForRelationExtraction.from_pretrained("/home/kewen_yang/Information_Extraction/RE_HuggingFace/model/checkpoint_2023_07_11_15_49.pt/checkpoint-5000")
  # relation_extraction_model = LayoutLMForRelationExtraction.from_pretrained("nielsr/layoutxlm-finetuned-xfund-fr-re")
  relation_extraction_model.to(DEVICE)

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

elif task.name == "XTRACT_INFER":
  """do inference by Xtract customized pipeline
  """
  print("initiating inferencing ...")
  model_dir = "/home/kewen_yang/Information_Extraction/RE_HuggingFace/model/checkpoint_2023_08_01_15_26.pt"
  relation_extraction_model = LayoutLMForRelationExtraction.from_pretrained(model_dir)
  relation_extraction_model.to(DEVICE)
  tokenizer = LayoutLMTokenizer.from_pretrained(model_dir)

  for file_name in os.listdir("/home/kewen_yang/Information_Extraction/dataset/Xtract_json_Batch_3"):

    # file_name = "10.json"

    with open(ROOT / "dataset/Xtract_json_Batch_3" / file_name, "rb") as f:
        file = json.load(f)

    entity_dict = file["entity_dict"]

    # entity_dict = {k: v[8:10] for k, v in entity_dict.items()}

    inputs = file["input"]

    #   del inputs["image"]  # image is only applicable to v2 model now

    print("---------------------------------------------------------------------------------------------")
    print("key-values before feeding to RE model:")
    print(f'questions: {[tokenizer.decode([i for i in inputs["input_ids"][0][s:e]]) for s, e, l in zip(entity_dict["start"], entity_dict["end"], entity_dict["label"]) if l == 1]}')
    print(f'answers: {[tokenizer.decode([i for i in inputs["input_ids"][0][s:e]]) for s, e, l in zip(entity_dict["start"], entity_dict["end"], entity_dict["label"]) if l == 2]}')
    print("---------------------------------------------------------------------------------------------")

    for k, v in inputs.items():
        inputs[k] = torch.tensor(inputs[k])

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    if len(entity_dict["label"]) == 0:
        pred_rel = []

    else:
        with torch.no_grad():
            # inputs: {'input_ids', 'token_type_ids', 'attention_mask', 'bbox'}
            outputs = relation_extraction_model(**inputs,
                                                entities=[entity_dict],
                                                relations=[{'start_index': [], 'end_index': [], 'head': [], 'tail': []}])
        pred_rel = outputs.pred_relations[0]

    res = defaultdict(list)
    print("---------------------------------------------------------------------------------------------")
    print("key-value pairs by RE model:")
    for relation in pred_rel:
        head_start, head_end = relation['head']
        tail_start, tail_end = relation['tail']
        key_d = {}
        key_d["id"] = relation["head_id"]
        key_d["text"] = tokenizer.decode(inputs['input_ids'][0][head_start:head_end])
        key_d["label"] = "question"
        key_d["bbox"] = torch.cat((inputs['bbox'][0][head_start:head_end].min(0).values[:2], inputs['bbox'][0][head_start:head_end].max(0).values[2:]), 0).tolist()
        key_d["linking"] = [[relation['head_id'], relation['tail_id']]]
        key_d["row_idx"] = entity_dict["row_idx"][entity_dict["start"].index(head_start)]
        res[relation["head_id"]].append(key_d)
        print(f"Question: {key_d['text']}")

        val_d = {}
        val_d["id"] = relation["tail_id"]
        val_d["text"] = tokenizer.decode(inputs['input_ids'][0][tail_start:tail_end])
        val_d["label"] = "answer"
        val_d["bbox"] = torch.cat((inputs['bbox'][0][tail_start:tail_end].min(0).values[:2], inputs['bbox'][0][tail_start:tail_end].max(0).values[2:]), 0).tolist()
        val_d["linking"] = [[relation['head_id'], relation['tail_id']]]
        val_d["row_idx"] = entity_dict["row_idx"][entity_dict["start"].index(tail_start)]
        res[relation["tail_id"]].append(val_d)
        print(f"Answer:, {val_d['text']}")
        print("-------------------------")

    # remove duplicates
    def concat_links(lst):
        """concantenate all the links for an entity

        Args:
            lst: list of entities with same id, to be concatenated

        Returns:
            Dict: entity
        """
        if len(lst) == 1:
            return lst[0]

        out = lst.pop(0)
        for e in lst:
            out["linking"].append(copy.deepcopy(e["linking"][0]))

        return out

    def choose_qentity(qns, ans):
        if len(qns) == 1:
            return None

        qns = sorted(qns, key=lambda q: (abs(q["row_idx"] - ans["row_idx"]), abs(ans["bbox"][0] - q["bbox"][2])))

        return qns[0]

    res = {k: concat_links(v) for k, v in res.items()}

    # remove redundant links based on distance
    for i in res.keys():
        if res[i]['label'] == "question":
            continue

        all_qns = [res[l[0]] for l in res[i]["linking"]]
        qns = choose_qentity(all_qns, res[i])
        if qns is not None:
            res[i]["linking"] = [[int(qns["id"]), int(res[i]["id"])]]
            res[qns["id"]]["linking"] = [[int(qns["id"]), int(res[i]["id"])]]

    for i in res.keys():
        if res[i]["label"] == "question" and len(res[i]["linking"]) > 1:

            drops = list()
            for l in res[i]["linking"]:
                if res[l[1]]["linking"][0][0] != i:
                    drops.append(l)

            if drops:
                for l in drops:
                    res[i]["linking"].remove(l)

    res = list(res.values())

    keyvalue, unpaired = post_process_entities(res)

    out = {"keyvalue": keyvalue, "unpaired": unpaired}

    # print key value text
    printable = {e["id"]: e for e in out["keyvalue"]}
    print("---------------------------------------------------------------------------------------------")
    print("key-value pairs after post-processing:")
    for e in printable.values():
        if e["label"] == "question":
            print(f'{e["text"]}:{printable[e["linking"]]["text"]}')

    with open(ROOT / "dataset/RE_pred_Batch_3" / file_name, 'w') as f:
        json.dump(out, f, indent=2)
    print("---------------------------------------------------------------------------------------------")

################################################ Appendix ################################################
"""
Input Schema:

{
    "id": image name, e.g. "en_train_0_0"
    "input_ids": List[token_ids], all_texts = "".join([tokenizer.decode(i) for i in dataset["train"][0]["input_ids"]])
    "bbox": tensors,
    "labels": tensors,
    "hd_image":
    "entities":
    "relations":
    "attention_mask":
    "image":
}

for English, tokens usually equals to words, but when word is out of vacab, it will be tokenized into multiple common tokens. e.g. "vashering" --> ['va', '##sher', '##ing']


>>>>input feature:
{"input_ids": [], "bbox": []}
input_ids: the token ids
bbox: bbox
e.g.
# show all the tokens
[self.tokenizer.decode(i) for i in feature["input_ids"]]
>> ['2', 'w', '##m', '##f', 'w', '##m', '##f', 'consumer', 'electric', 'gmbh', 'mess', '##ers', '##ch', '##mit', ...]

# get the original texts
self.tokenizer.convert_tokens_to_string([self.tokenizer.decode(i) for i in feature["input_ids"]])
>> '2 wmf wmf consumer electric gmbh messerschmittstrabe : d - 89343 jettingen - scheppach wmf kult x mono induction hob art. nr. : 04 1524 8811 2x art. nr. : 04 _ 1524 _ 8811 ean : 421112 9145688 cmmf : 3200001681'


>>>>entities:
{"start": [], "end": [], "label": []}

"start": starting index of input_ids for this entity
"end": ending index of input_ids for this entity
"label": label

e.g.
ens = [self.tokenizer.convert_tokens_to_string([self.tokenizer.decode(i) for i in feature["input_ids"][s:e]]) for s, e in zip(entities[1]["start"], entities[1]["end"])]
>> ['art. nr. : 04', '04 1524 8811 2', 'art. nr. : 04', '04 _ 1524 _ 8811 ea', 'ean : 421', '421112 9145688 cm', 'cmmf : 320', '3200001681']


>>>>relations:
{"head": [], "tail": [], "start_index": [], "end_index" : []}

"start_index": starting index of input_ids for this Question-Answer pair
"end_index": ending index of input_ids for this Question-Answer pair

e.g.
[self.tokenizer.convert_tokens_to_string([self.tokenizer.decode(i) for i in feature["input_ids"][s:e]]) for s, e in zip(relations[1]["start_index"], relations[1]["end_index"])]
>> ['art. nr. : 04 1524 8811', 'art. nr. : 04 _ 1524 _ 8811', 'ean : 421112 9145688', 'cmmf : 3200001681']

"head": question index w.r.t the index of "entities"
"tail": answer index

e.g.
[(ens[h], ens[t]) for h, t in zip(relations[1]["head"], relations[1]["tail"])]
>> [('art. nr. :', '04 1524 8811'), ('art. nr. :', '04 _ 1524 _ 8811'), ('ean :', '421112 9145688'), ('cmmf :', '3200001681')]
"""