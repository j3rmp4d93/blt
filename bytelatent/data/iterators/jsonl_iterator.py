# Copyright (c) Meta Platforms, Inc. and affiliates.
import re
from logging import getLogger
from pathlib import Path
from typing import Any, Generator

import pyarrow as pa
import pyarrow.dataset as ds

# pyarrow needs the initialization from this import
from pydantic import BaseModel, ConfigDict

from bytelatent import ByteLatentError
from bytelatent.data.data_types import LMExample
from bytelatent.data.iterators.abstract_iterator import IteratorState, StatefulIterator

logger = getLogger(__name__)


class JsonlIteratorState(BaseModel, IteratorState):
    model_config = ConfigDict(extra="forbid")
    row_num: int
    num_workers: int
    worker_id: int
    dataset_files: list[str]
    arrow_batch_size: int = 100

    def build(self) -> "JsonlIterator":
        arrow_file = JsonlIterator(
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            arrow_batch_size=self.arrow_batch_size,
            dataset_files=self.dataset_files,
        )
        if self.row_num != 0:
            arrow_file._set_row_num(self.row_num)
        return arrow_file


def shard_sort_key(file: str | Path):
    match = re.search(r".+\.shard_([0-9]+)\.arrow", str(file))
    shard_number = int(match.group(1))
    return shard_number


class JsonlIterator(StatefulIterator):
    def __init__(
        self,
        *,
        dataset_files: list[str],
        worker_id: int,
        num_workers: int,
        arrow_batch_size: int,
    ):
        assert 0 <= worker_id < num_workers, (worker_id, num_workers)
        self.row_num = 0
        self.iter_id = 0
        self.batch_iterator = None
        self.batch_to_consume = None
        self.dataset = None
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.arrow_batch_size = arrow_batch_size
        self.dataset_files = dataset_files

    def get_state(self) -> JsonlIteratorState:
        return JsonlIteratorState(
            row_num=self.row_num,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            arrow_batch_size=self.arrow_batch_size,
            dataset_files=self.dataset_files,
        )

    def create_iter(
        self,
    ) -> Generator[LMExample, Any, None]:
        if self.dataset is None:
            self.dataset = ds.dataset(self.dataset_files, format="json")
            self.batch_iterator = self.dataset.to_batches(
                batch_size=self.arrow_batch_size
            )
        self.iter_id += 1
        if self.batch_to_consume is not None:
            batch_columns: dict[str, list] = self.batch_to_consume
            self.batch_to_consume = None
            sample_ids = batch_columns["id"]
            texts = batch_columns["text"]
            for i in range(len(sample_ids)):
                out = LMExample(
                    sample_id=sample_ids[i],
                    text=texts[i],
                    tokens=None,
                    mask=None,
                )
                self.row_num += 1
                if (self.row_num - 1) % self.num_workers == self.worker_id:
                    yield out

        for batch in self.batch_iterator:
            batch_columns = batch.to_pydict()
            sample_ids = batch_columns["id"]
            texts = batch_columns["text"]
            for i in range(len(sample_ids)):
                out = LMExample(
                    sample_id=sample_ids[i],
                    text=texts[i],
                    tokens=None,
                    mask=None,
                )
                self.row_num += 1
                if (self.row_num - 1) % self.num_workers == self.worker_id:
                    yield out

    def _set_row_num(self, target_row_num: int):
        logger.info(
            f"Setting arrow position to {target_row_num} for {self.dataset_files}"
        )
        if target_row_num is None or target_row_num == 0:
            self.row_num = 0
            self.dataset = None
            self.batch_iterator = None
            self.batch_to_consume = None
        else:
            self.dataset = pa.dataset.dataset(self.dataset_files, format="arrow")
            self.batch_iterator = self.dataset.to_batches(
                batch_size=self.arrow_batch_size
            )
            curr_remaining = target_row_num
            for batch in self.batch_iterator:
                if len(batch) > curr_remaining:
                    batch_columns: dict[str, list] = batch.to_pydict()
                    batch_columns["sample_id"] = batch_columns["sample_id"][
                        curr_remaining:
                    ]
                    batch_columns["entropies"] = batch_columns["entropies"][
                        curr_remaining:
                    ]
                    batch_columns["text"] = batch_columns["text"][curr_remaining:]
                    self.batch_to_consume = batch_columns
                    break
                elif len(batch) == curr_remaining:
                    # We are exactly at the end of the batch,
                    # so the next batch is the right spot
                    break
                else:
                    curr_remaining -= len(batch)
            self.row_num = target_row_num
        logger.info(
            f"Finished setting arrow position to {target_row_num} for {self.dataset_files}"
        )


TRAIN_DATA_FILE_PATTERN = "*.chunk.*.jsonl"


def find_and_sanitize_chunks(
    dataset_path: str, world_size: int, file_pattern: str = TRAIN_DATA_FILE_PATTERN
):
    dataset_chunks = [str(p) for p in Path(dataset_path).glob(file_pattern)]
    n_chunks = len(dataset_chunks)

    if n_chunks > world_size:
        n_discard = n_chunks - world_size
        dataset_chunks = dataset_chunks[:world_size]
    else:
        assert (
            world_size % n_chunks == 0
        ), "World size should be a multiple of number of chunks"

    assert n_chunks > 0, f"No valid chunks in {dataset_path}"

    return dataset_chunks



