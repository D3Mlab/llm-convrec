from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from tensorflow import keras
from keras import layers
import torch
import transformers
import numpy as np
from transformers.models.distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast

transformers.logging.set_verbosity_error()

"""
   Modified based on  https://github.com/D3Mlab/rir/blob/main/prefernce_matching/LM.py
"""


class BERT_model:
    _bert_name: str
    _tokenizer: DistilBertTokenizerFast
    _bert_model: keras.Model
    _first_input: str
    _second_input: str
    _device: torch.device

    def __init__(self, bert_name: str, tokenizer_name: str, from_pt: bool = True):
        """
        :param bert_name: name or address of language prefernce_matching
        :param tokenizer_name: name or address of the tokenizer
        """
        self._bert_name = bert_name
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._bert_model, self._first_input, self._second_input = self._create_model(bert_name, from_pt)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def embed(self, texts: list[str], strategy=None, bs=48, verbose=0) -> np.ndarray:
        """
        Embed the batch of texts.

        :param texts: list of strings to be embedded
        :param strategy: Defaults to None.
        :param bs: Defaults to 48.
        :param verbose: Defaults to 0.
        :return: embeddings of texts
        """
        tokenized_review = self._tokenizer.batch_encode_plus(
            texts,
            max_length=512,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
        )

        data = {
            self._first_input: tokenized_review['input_ids'],
            self._second_input: tokenized_review['attention_mask'],
        }

        if strategy is not None:
            with strategy.scope():
                dataset = tf.data.Dataset.from_tensor_slices(data).batch(bs, drop_remainder=False).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
                outputs = self._bert_model.predict(dataset, verbose=verbose)
                return outputs['last_hidden_state'][:, 0, :].reshape(-1, 768)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(data).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE).batch(bs, drop_remainder=False)
            outputs = self._bert_model.predict(dataset, verbose=verbose)
            return outputs['last_hidden_state'][:, 0, :].reshape(-1, 768)

    def get_tensor_embedding(self, query: str) -> torch.Tensor:
        """
        Get a tensor embedding of a string.

        :param query: string to be embedded
        :return: tensor embedding of query
        """
        query_embedding = self.embed([query])
        query_embedding = torch.tensor(query_embedding).to(self._device)
        query_embedding = query_embedding.squeeze(0)

        return query_embedding

    def _create_model(self, bert_name: str, from_pt: bool = True) -> tuple[keras.Model, str, str]:
        # BERT encoder
        encoder = TFAutoModel.from_pretrained(bert_name, from_pt=from_pt)

        # Model
        input_ids = layers.Input(shape=(None,), dtype=tf.int32)
        attention_mask = layers.Input(shape=(None,), dtype=tf.int32)

        embedding = encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        model = keras.Model(
            inputs=[input_ids, attention_mask],
            outputs=embedding)

        model.compile()
        return model, input_ids.name, attention_mask.name
