# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from typing import Any, Dict, Mapping, Optional, Union

from torchtune.data._messages import Message
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


class MyDatasetToMessages(Transform):
    """
    A transform class for creating messages using a template filled with sample fields.

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt. Default is True.
        template (Dict[str, str]): A dictionary of templates with placeholders for dynamic fields.
            Example: {"prompt_input": "{instruction}\n\nInput: {input}"}
    """

    def __init__(
        self,
        train_on_input: bool = True,
        template: Dict[str, str] = None,
    ):
        self.train_on_input = train_on_input
        self.template = template or {"prompt_input": ""}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        # Use all fields in the sample for template filling
        prompt_template = self.template.get("prompt_input", "")
        if prompt_template != "":  # Fine-tuning mode: Use template for generating prompts
            try:
                prompt = prompt_template.format(**sample)
            except KeyError as e:
                missing_field = e.args[0]
                raise ValueError(
                    f"Missing field '{missing_field}' in the sample for template filling."
                ) from e

            messages = [
                Message(
                    role="system",
                    content="You are a helpful and harmless assistant.",
                    masked=True,
                    eot=True,
                ),
                Message(
                    role="user",
                    content=prompt,
                    masked=not self.train_on_input,
                    eot=True,
                ),
                Message(
                    role="assistant",
                    content=sample['response'],
                    masked=False,
                    eot=True,
                ),
            ]
        else: # Pretraining mode: role set with None
            messages = [
                Message(
                    role="",
                    content=sample['response'],
                    masked=False,
                    eot=True,
                ),
            ]
        return {"messages": messages}


def my_dataset(
    tokenizer: ModelTokenizer,
    *,
    template_str: str = None,
    source: str = "json", 
    data_file: str = None,
    train_on_input: bool = True,
    packed: bool = False,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    A flexible dataset loader for local JSON files or remote datasets using Hugging Face Datasets.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag:
    - If ``train_on_input`` is True, the prompt is used during training and contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100).

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        template_str (str): Type of template for dynamically formatting prompts. Keys define prompt structure.
        source (str): Dataset format (e.g., "json", "csv") for Hugging Face's `load_dataset`. Default is "json".
        data_file (str): Paths to local dataset file. Required if using a local dataset. 
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g., ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): Additional keyword arguments to pass to ``load_dataset``.

    Returns:
        Union[SFTDataset, PackedDataset]: Dataset configured with source data and transform.

    Raises:
        ValueError: If ``packed`` is True and ``max_seq_len`` is not set on the tokenizer.

    Example:
        >>> ds = my_dataset(
        ...     tokenizer=tokenizer,
        ...     data_files="/path/to/data.json",
        ...     train_on_input=True,
        ...     split="train",
        ... )
        >>> for batch in DataLoader(ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """
    if template_str:
        template = {"prompt_input": "{input}"}
    else:
        template = None
        
    # Validate data_files argument for local datasets
    if data_file is None:
        raise ValueError("When using a local dataset, 'data_file' must be specified.")
    
    # Create message transform
    message_transform = MyDatasetToMessages(
        train_on_input=train_on_input,
        template=template,
    )

    ds = SFTDataset(
        source=source,
        data_files=data_file, 
        message_transform=message_transform,
        model_transform=tokenizer,
        split=split,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
