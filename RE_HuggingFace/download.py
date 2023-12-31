# Lint as: python3
import json
import logging
import os
from pathlib import Path

import datasets

from PIL import Image
import numpy as np
import torch

# TODO: log msg in file

ROOT = Path(__file__).parents[1]

from transformers import AutoTokenizer


def load_image(image_path, size=None):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if size is not None:
        # resize image
        image = image.resize((size, size))
        image = np.asarray(image)
        image = image[:, :, ::-1] # flip color channels from RGB to BGR
        image = image.transpose(2, 0, 1) # move channels to first dimension
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


logger = logging.getLogger(__name__)


class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN."""

    def __init__(self, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XFUNConfig, self).__init__(**kwargs)


class XFUN(datasets.GeneratorBasedBuilder):
    """XFUN dataset."""

    BUILDER_CONFIGS = [XFUNConfig(name="en")]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.ClassLabel(
                            names=["O", "B-QUESTION", "B-ANSWER", "I-ANSWER", "I-QUESTION"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "original_image": datasets.features.Image(),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER"]),
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "head": datasets.Value("int64"),
                            "tail": datasets.Value("int64"),
                            "start_index": datasets.Value("int64"),
                            "end_index": datasets.Value("int64"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        file_dir = "/home/kewen_yang/Information_Extraction/dataset/RE_Finetune_1/"
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": [file_dir + "train.json",
                                                                    file_dir]}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": [file_dir + "val.json",
                                                                         file_dir]}),

            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]

    def _generate_examples(self, filepath):
        logger.info("Generating examples from = %s", filepath)
        with open(filepath[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        for doc in data["documents"]:
            try:
                doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
                size = (doc["img"]["width"], doc["img"]["height"])
                image, _ = load_image(doc["img"]["fpath"], size=224)
                original_image, _ = load_image(doc["img"]["fpath"])
                document = doc["document"]
                tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
                entities = []
                relations = []
                id2label = {}
                entity_id_to_index_map = {}
                empty_entity = set()
                for line in document:
                    if len(line["text"]) == 0:
                        empty_entity.add(line["id"])
                        continue
                    id2label[line["id"]] = line["label"]
                    relations.extend([tuple(sorted(l)) for l in line["linking"]])
                    tokenized_inputs = self.tokenizer(
                        line["text"],
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                    )
                    text_length = 0
                    ocr_length = 0
                    bbox = []
                    for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                        if token_id == 6:
                            bbox.append(None)
                            continue
                        text_length += offset[1] - offset[0]
                        tmp_box = []
                        while ocr_length < text_length:
                            ocr_word = line["words"].pop(0)
                            ocr_length += len(
                                self.tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                            )
                            tmp_box.append(simplify_bbox(ocr_word["box"]))
                        if len(tmp_box) == 0:
                            tmp_box = last_box
                        bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                        last_box = tmp_box  # noqa
                    bbox = [
                        [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                        for i, b in enumerate(bbox)
                    ]

                    # remove header from labels
                    if line["label"] in ["other", "header"]:
                        label = ["O"] * len(bbox)
                    else:
                        label = [f"I-{line['label'].upper()}"] * len(bbox)
                        label[0] = f"B-{line['label'].upper()}"

                    tokenized_inputs.update({"bbox": bbox, "labels": label})
                    if label[0] != "O":
                        entity_id_to_index_map[line["id"]] = len(entities)
                        entities.append(
                            {
                                "start": len(tokenized_doc["input_ids"]),
                                "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                                "label": line["label"].upper(),
                            }
                        )
                    for i in tokenized_doc:
                        tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]

                relations = list(set(relations))
                relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
                kvrelations = []
                for rel in relations:
                    pair = [id2label[rel[0]], id2label[rel[1]]]
                    if pair == ["question", "answer"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                        )
                    elif pair == ["answer", "question"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                        )
                    else:
                        continue

                def get_relation_span(rel):
                    bound = []
                    for entity_index in [rel["head"], rel["tail"]]:
                        bound.append(entities[entity_index]["start"])
                        bound.append(entities[entity_index]["end"])
                    return min(bound), max(bound)

                relations = sorted(
                    [
                        {
                            "head": rel["head"],
                            "tail": rel["tail"],
                            "start_index": get_relation_span(rel)[0],
                            "end_index": get_relation_span(rel)[1],
                        }
                        for rel in kvrelations
                    ],
                    key=lambda x: x["head"],
                )
                chunk_size = 512
                for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
                    item = {}
                    for k in tokenized_doc:
                        item[k] = tokenized_doc[k][index: index + chunk_size]
                    entities_in_this_span = []
                    global_to_local_map = {}
                    for entity_id, entity in enumerate(entities):
                        if (
                            index <= entity["start"] < index + chunk_size
                            and index <= entity["end"] < index + chunk_size
                        ):
                            entity["start"] = entity["start"] - index
                            entity["end"] = entity["end"] - index
                            global_to_local_map[entity_id] = len(entities_in_this_span)
                            entities_in_this_span.append(entity)
                    relations_in_this_span = []
                    for relation in relations:
                        if (
                            index <= relation["start_index"] < index + chunk_size
                            and index <= relation["end_index"] < index + chunk_size
                        ):
                            relations_in_this_span.append(
                                {
                                    "head": global_to_local_map[relation["head"]],
                                    "tail": global_to_local_map[relation["tail"]],
                                    "start_index": relation["start_index"] - index,
                                    "end_index": relation["end_index"] - index,
                                }
                            )
                    # check if any bbox value > 1000
                    if max(torch.tensor(item["bbox"])[:, 1]) > 1000:
                        raise ValueError(doc["img"]["fpath"])

                    item.update(
                        {
                            "id": f"{doc['id']}_{chunk_id}",
                            "image": image,
                            "original_image": original_image,
                            "entities": entities_in_this_span,
                            "relations": relations_in_this_span,
                        }
                    )
                    yield f"{doc['id']}_{chunk_id}", item
            except:
                print(f'>>>>>>>>>>>>>>problem with {doc["id"]}')


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
"""