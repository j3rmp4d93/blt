# Copyright (c) Meta Platforms, Inc. and affiliates.
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import typer
from rich.progress import Progress, TextColumn

from bytelatent.data.iterators.jsonl_iterator import JsonlIterator


def main(
    input_file: str,
    output_file: str,
    log_step: int = 10_000,
    dry_run: bool = False,
):
    # TODO: Modify this to work with the new code
    iterator = JsonlIterator(
        dataset_files=[input_file],
        worker_id=0,
        num_workers=1,
        arrow_batch_size=100,
    ).create_iter()
    print(f"Preprocessing, input: {input_file}, output: {output_file}")
    if dry_run:
        return

    step = 0
    print("starting")
    start_time = time.time()
    sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
    text_field = pa.field("text", pa.string(), nullable=False)
    schema = pa.schema([sample_id_field, text_field])
    arrow_batch_size = 1_000

    try:
        with pa.OSFile(output_file, "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                id_buffer = []
                text_buffer = []
                with Progress(
                    *Progress.get_default_columns(),
                    TextColumn("Completed: {task.completed}"),
                ) as progress:
                    task = progress.add_task(
                        "[green]Writing arrow files...", total=None
                    )
                    for doc in iterator:
                        sample_id = doc.sample_id
                        text = doc.text                    
                        id_buffer.append(sample_id)
                        text_buffer.append(text)
                        if len(id_buffer) == arrow_batch_size:
                            batch = pa.record_batch(
                                {
                                    "sample_id": id_buffer,
                                    "text": text_buffer,
                                },
                                schema,
                            )
                            writer.write(batch)
                            id_buffer = []
                            text_buffer = []
                        step += 1
                        if step % log_step == 0:
                            print("Completed steps:", step)
                        progress.update(task, advance=1)
                    if len(id_buffer) > 0:
                        # Write last things
                        batch = pa.record_batch(
                            {
                                "sample_id": id_buffer,
                                "text": text_buffer,
                            },
                            schema,
                        )
                        writer.write(batch)
                        id_buffer = []
                        text_buffer = []
        Path(f"{output_file}.complete").touch()
    except:
        Path(output_file).unlink(missing_ok=True)
        raise
    elapsed = time.time() - start_time
    print("steps", step)
    print("done in:", elapsed)


if __name__ == "__main__":
    typer.run(main)
