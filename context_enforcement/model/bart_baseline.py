# PARAMETERS BART BASE
# ==============================================================================
from typing import Optional

from transformers import BartConfig, BartForConditionalGeneration

MAX_POSITION_EMBEDDINGS = 1024
ENCODER_LAYERS = 6
ENCODER_FFN_DIM = 3072
ENCODER_ATTENTION_HEADS = 12
DECODER_LAYERS = 6
DECODER_FFN_DIM = 3072
DECODER_ATTENTION_HEADS = 12
D_MODEL = 768
DROPOUT = 0.1


# ==============================================================================
# PARAMETERS

def create_bart_baseline(vocab_size,
                         max_seq_len,
                         pad_token_id,
                         bos_token_id,
                         eos_token_id,
                         encoder_layers: Optional[int] = None,
                         encoder_ffn_dim: Optional[int] = None,
                         encoder_attention_heads: Optional[int] = None,
                         decoder_layers: Optional[int] = None,
                         decoder_ffn_dim: Optional[int] = None,
                         decoder_attention_heads: Optional[int] = None,
                         d_model: Optional[int] = None,
                         dropout: Optional[float] = None,
                         )->BartForConditionalGeneration:
    """
    Instantiates the baseline version of the BART model

    :param vocab_size:
    :param max_seq_len:
    :param pad_token_id:
    :param bos_token_id:
    :param eos_token_id:
    :param encoder_layers:
    :param encoder_ffn_dim:
    :param encoder_attention_heads:
    :param decoder_layers:
    :param decoder_ffn_dim:
    :param decoder_attention_heads:
    :param d_model:
    :param dropout:
    :return:
    """

    if encoder_layers is None:
        encoder_layers = ENCODER_LAYERS
    if encoder_ffn_dim is None:
        encoder_ffn_dim = ENCODER_FFN_DIM
    if encoder_attention_heads is None:
        encoder_attention_heads = ENCODER_ATTENTION_HEADS
    if decoder_layers is None:
        decoder_layers = DECODER_LAYERS
    if decoder_ffn_dim is None:
        decoder_ffn_dim = DECODER_FFN_DIM
    if decoder_attention_heads is None:
        decoder_attention_heads = DECODER_ATTENTION_HEADS
    if d_model is None:
        d_model = D_MODEL
    if dropout is None:
        dropout = DROPOUT

    model = BartForConditionalGeneration(
        BartConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
            encoder_layers=encoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_attention_heads=encoder_attention_heads,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_attention_heads=decoder_attention_heads,
            d_model=d_model,
            dropout=dropout,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=True,
            decoder_start_token_id=eos_token_id,
        )
    )
    return model
