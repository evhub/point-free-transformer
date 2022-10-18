#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xbcf40d75

# Compiled with Coconut version 2.0.0-post_dev15 [How Not to Be Seen]

# Coconut Header: -------------------------------------------------------------

from __future__ import generator_stop
import sys as _coconut_sys, os as _coconut_os
_coconut_file_dir = _coconut_os.path.dirname(_coconut_os.path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get("__coconut__")
if _coconut_cached_module is not None and _coconut_os.path.dirname(_coconut_cached_module.__file__) != _coconut_file_dir:  # type: ignore
    del _coconut_sys.modules["__coconut__"]
_coconut_sys.path.insert(0, _coconut_file_dir)
_coconut_module_name = _coconut_os.path.splitext(_coconut_os.path.basename(_coconut_file_dir))[0]
if _coconut_module_name and _coconut_module_name[0].isalpha() and all(c.isalpha() or c.isdigit() for c in _coconut_module_name) and "__init__.py" in _coconut_os.listdir(_coconut_file_dir):
    _coconut_full_module_name = str(_coconut_module_name + ".__coconut__")
    import __coconut__ as _coconut__coconut__
    _coconut__coconut__.__name__ = _coconut_full_module_name
    for _coconut_v in vars(_coconut__coconut__).values():
        if getattr(_coconut_v, "__module__", None) == "__coconut__":
            try:
                _coconut_v.__module__ = _coconut_full_module_name
            except AttributeError:
                _coconut_v_type = type(_coconut_v)
                if getattr(_coconut_v_type, "__module__", None) == "__coconut__":
                    _coconut_v_type.__module__ = _coconut_full_module_name
    _coconut_sys.modules[_coconut_full_module_name] = _coconut__coconut__
from __coconut__ import *
from __coconut__ import _coconut_tail_call, _coconut_tco, _namedtuple_of, _coconut, _coconut_super, _coconut_MatchError, _coconut_iter_getitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_raise, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec, _coconut_comma_op, _coconut_multi_dim_arr, _coconut_mk_anon_namedtuple, _coconut_matmul
_coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

from typing import TypeVar  #1 (line in Coconut source)
from typing import Any  #1 (line in Coconut source)
try:  #1 (line in Coconut source)
    _coconut_sys_0 = sys  #1 (line in Coconut source)
except _coconut.NameError:  #1 (line in Coconut source)
    _coconut_sys_0 = _coconut_sentinel  #1 (line in Coconut source)
sys = _coconut_sys  #1 (line in Coconut source)
if sys.version_info >= (3, 8):  #1 (line in Coconut source)
    from typing import Literal  #1 (line in Coconut source)
else:  #1 (line in Coconut source)
    from typing_extensions import Literal  #1 (line in Coconut source)
if _coconut_sys_0 is not _coconut_sentinel:  #1 (line in Coconut source)
    sys = _coconut_sys_0  #1 (line in Coconut source)

import numpy as np  #7 (line in Coconut source)
from scipy.stats import norm  # type: ignore  #8 (line in Coconut source)
from scipy.special import softmax  # type: ignore  #9 (line in Coconut source)
from better_einsum import einsum  #10 (line in Coconut source)


# Constants:

_S = TypeVar("_S", bound=tuple[int, ...])  #15 (line in Coconut source)
SizedArr = np.ndarray[_S, np.dtype[Any]]  #16 (line in Coconut source)

N = TypeVar("N", bound=int)  #18 (line in Coconut source)
M = TypeVar("M", bound=int)  #19 (line in Coconut source)
L = TypeVar("L", bound=int)  #20 (line in Coconut source)

from_literal: '_coconut.typing.Callable[[Any], int]' = _coconut_base_compose(_coconut.operator.attrgetter("__args__"), (_coconut.operator.itemgetter((0)), 0))  #22 (line in Coconut source)

N_vocab = Literal[1024]  #24 (line in Coconut source)
n_vocab: 'int' = from_literal(N_vocab)  #25 (line in Coconut source)

N_seq = Literal[64]  #27 (line in Coconut source)
n_seq: 'int' = from_literal(N_seq)  #28 (line in Coconut source)

N_model = Literal[256]  #30 (line in Coconut source)
n_model: 'int' = from_literal(N_model)  #31 (line in Coconut source)

N_heads = Literal[8]  #33 (line in Coconut source)
n_heads: 'int' = from_literal(N_heads)  #34 (line in Coconut source)

N_head_size = Literal[32]  #36 (line in Coconut source)
n_head_size: 'int' = from_literal(N_head_size)  #37 (line in Coconut source)
assert n_head_size == n_model // n_heads  #38 (line in Coconut source)

N_ff = Literal[128]  #40 (line in Coconut source)
n_ff: 'int' = from_literal(N_ff)  #41 (line in Coconut source)

eps: 'float' = 1e-5  #43 (line in Coconut source)


# Weights:

W_enc: 'SizedArr[_coconut.typing.Tuple[N_vocab, N_model]]' = np.random.normal(size=(n_vocab, n_model))  #48 (line in Coconut source)
W_pos: 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]' = np.random.normal(size=(n_seq, n_model))  #49 (line in Coconut source)

W_unenc: 'SizedArr[_coconut.typing.Tuple[N_model, N_vocab]]' = np.random.normal(size=(n_model, n_vocab))  #51 (line in Coconut source)

ln_gain_unenc: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #53 (line in Coconut source)
ln_bias_unenc: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #54 (line in Coconut source)

mask: 'SizedArr[_coconut.typing.Tuple[N_seq, N_seq, Literal[1]]]' = np.tril(np.ones((n_seq, n_seq)))[..., None]  #56 (line in Coconut source)

sample_temp: 'float' = 1.0  #58 (line in Coconut source)

ln_gain_attn: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #60 (line in Coconut source)
ln_bias_attn: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #61 (line in Coconut source)

W_Q: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]' = np.random.normal(size=(n_model, n_model))  #63 (line in Coconut source)
b_Q: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #64 (line in Coconut source)

W_K: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]' = np.random.normal(size=(n_model, n_model))  #66 (line in Coconut source)
b_K: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #67 (line in Coconut source)

W_V: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]' = np.random.normal(size=(n_model, n_model))  #69 (line in Coconut source)
b_V: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #70 (line in Coconut source)

W_O: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]' = np.random.normal(size=(n_model, n_model))  #72 (line in Coconut source)
b_O: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #73 (line in Coconut source)

ln_gain_ff: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #75 (line in Coconut source)
ln_bias_ff: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #76 (line in Coconut source)

W_ff_in: 'SizedArr[_coconut.typing.Tuple[N_model, N_ff]]' = np.random.normal(size=(n_model, n_ff))  #78 (line in Coconut source)
b_ff_in: 'SizedArr[_coconut.typing.Tuple[N_ff,]]' = np.random.normal(size=(n_ff,))  #79 (line in Coconut source)

W_ff_out: 'SizedArr[_coconut.typing.Tuple[N_ff, N_model]]' = np.random.normal(size=(n_ff, n_model))  #81 (line in Coconut source)
b_ff_out: 'SizedArr[_coconut.typing.Tuple[N_model,]]' = np.random.normal(size=(n_model,))  #82 (line in Coconut source)


# Point-free one layer transformer:

base_layer_norm = lift(_coconut.operator.truediv)(lift(_coconut_minus)(ident, _coconut.operator.methodcaller("mean", axis=-1, keepdims=True)), (_coconut_base_compose(_coconut.operator.methodcaller("var", axis=-1, keepdims=True), ((_coconut_partial(_coconut.operator.add, {1: eps}, 2, ())), 0))))  # type: ignore  #87 (line in Coconut source)

point_free_one_layer_transformer = (_coconut_base_compose((_coconut_base_compose((_coconut_partial(_coconut_matmul, {1: W_enc}, 2, ())), ((_coconut_partial(_coconut.operator.truediv, {1: n_model**0.5}, 2, ())), 0), ((_coconut_partial(_coconut.operator.add, {1: W_pos}, 2, ())), 0))), ((_coconut_base_compose((_coconut_base_compose(base_layer_norm, ((_coconut_partial(_coconut.operator.mul, {1: ln_gain_attn}, 2, ())), 0), ((_coconut_partial(_coconut.operator.add, {1: ln_bias_attn}, 2, ())), 0))), (lift(_coconut.operator.add)(ident, (_coconut_base_compose(lift(_coconut.functools.partial(einsum, "residual[q,h,d] = attn[q,k,h] * V[k,h,d]"))(attn=(_coconut_base_compose(lift(_coconut.functools.partial(einsum, "attn[q,k,h] = Q[q,h,d] * K[k,h,d]"))(Q=(_coconut_base_compose((_coconut_partial(_coconut_matmul, {1: W_Q}, 2, ())), ((_coconut_partial(_coconut.operator.add, {1: b_Q}, 2, ())), 0), (_coconut.operator.methodcaller("reshape", n_seq, n_heads, n_head_size), 0))), K=(_coconut_base_compose((_coconut_partial(_coconut_matmul, {1: W_K}, 2, ())), ((_coconut_partial(_coconut.operator.add, {1: b_K}, 2, ())), 0), (_coconut.operator.methodcaller("reshape", n_seq, n_heads, n_head_size), 0)))), ((_coconut_partial(_coconut.operator.truediv, {1: n_head_size**0.5}, 2, ())), 0), ((_coconut_partial(_coconut.operator.add, {1: np.where(mask, 0, float("-inf"))}, 2, ())), 0), (_coconut.functools.partial(softmax, axis=1), 0))), V=(_coconut_base_compose((_coconut_partial(_coconut_matmul, {1: W_V}, 2, ())), ((_coconut_partial(_coconut.operator.add, {1: b_V}, 2, ())), 0), (_coconut.operator.methodcaller("reshape", n_seq, n_heads, n_head_size), 0)))), (_coconut.operator.methodcaller("reshape", n_seq, n_model), 0), ((_coconut_partial(_coconut_matmul, {1: W_O}, 2, ())), 0), ((_coconut_partial(_coconut.operator.add, {1: b_O}, 2, ())), 0)))), 0), ((_coconut_base_compose(base_layer_norm, ((_coconut_partial(_coconut.operator.mul, {1: ln_gain_ff}, 2, ())), 0), ((_coconut_partial(_coconut.operator.add, {1: ln_bias_ff}, 2, ())), 0))), 0), (lift(_coconut.operator.add)(ident, (_coconut_base_compose((_coconut_partial(_coconut_matmul, {1: W_ff_in}, 2, ())), ((_coconut_partial(_coconut.operator.add, {1: b_ff_in}, 2, ())), 0), (lift(_coconut.operator.mul)(ident, norm.cdf), 0), ((_coconut_partial(_coconut_matmul, {1: W_ff_out}, 2, ())), 0), ((_coconut_partial(_coconut.operator.add, {1: b_ff_out}, 2, ())), 0)))), 0))), 0), ((_coconut_base_compose(base_layer_norm, ((_coconut_partial(_coconut.operator.mul, {1: ln_gain_unenc}, 2, ())), 0), ((_coconut_partial(_coconut.operator.add, {1: ln_bias_unenc}, 2, ())), 0))), 0), ((_coconut_partial(_coconut_matmul, {1: W_unenc}, 2, ())), 0), ((_coconut_base_compose((_coconut_partial(_coconut.operator.truediv, {1: sample_temp}, 2, ())), (_coconut.functools.partial(softmax, axis=-1), 0))), 0)))  # type: ignore  #98 (line in Coconut source)


# Reference implementation:

@_coconut_tco  #180 (line in Coconut source)
def transformer_model(transformer_layers: '_coconut.typing.Sequence[(_coconut.typing.Callable[[SizedArr[_coconut.typing.Tuple[N_seq, N_model]]], SizedArr[_coconut.typing.Tuple[N_seq, N_model]]])]', one_hot_tokens: 'SizedArr[_coconut.typing.Tuple[N_seq, N_vocab]]',) -> 'SizedArr[_coconut.typing.Tuple[N_seq, N_vocab]]':  #180 (line in Coconut source)
    residual: 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]' = embed(one_hot_tokens)  #186 (line in Coconut source)
    for layer in transformer_layers:  #187 (line in Coconut source)
        residual = (layer)(residual)  #188 (line in Coconut source)
    residual = (layer_norm)(ln_gain_unenc, ln_bias_unenc, residual)  #189 (line in Coconut source)
    logits: 'SizedArr[_coconut.typing.Tuple[N_seq, N_vocab]]' = residual @ W_unenc  #190 (line in Coconut source)
    return _coconut_tail_call(sample_probs, logits)  #191 (line in Coconut source)


@_coconut_tco  #193 (line in Coconut source)
def sample_probs(logits: 'SizedArr[_coconut.typing.Tuple[N_seq, N_vocab]]',) -> 'SizedArr[_coconut.typing.Tuple[N_seq, N_vocab]]':  #193 (line in Coconut source)
    logits /= sample_temp  #196 (line in Coconut source)
    return _coconut_tail_call((softmax), logits, axis=-1)  #197 (line in Coconut source)


def layer_norm(gain: 'SizedArr[_coconut.typing.Tuple[N_model,]]', bias: 'SizedArr[_coconut.typing.Tuple[N_model,]]', residual: 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]',) -> 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]':  #199 (line in Coconut source)
    residual -= residual.mean(axis=-1, keepdims=True)  #204 (line in Coconut source)
    residual /= residual.var(axis=-1, keepdims=True) + eps  #205 (line in Coconut source)
    return (residual * gain + bias)  #206 (line in Coconut source)


def embed(one_hot_tokens: 'SizedArr[_coconut.typing.Tuple[N_seq, N_vocab]]',) -> 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]':  #208 (line in Coconut source)
    return (one_hot_tokens @ W_enc / n_model**0.5 + W_pos)  #211 (line in Coconut source)


@_coconut_tco  #213 (line in Coconut source)
def attn_prep(W: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]', b: 'SizedArr[_coconut.typing.Tuple[N_model,]]', residual: 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]',) -> 'SizedArr[_coconut.typing.Tuple[N_seq, N_heads, N_head_size]]':  #213 (line in Coconut source)
    residual = residual @ W + b  #218 (line in Coconut source)
    return _coconut_tail_call(residual.reshape, n_seq, n_heads, n_head_size)  #219 (line in Coconut source)


def multi_head_attn(W_Q: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]', b_Q: 'SizedArr[_coconut.typing.Tuple[N_model,]]', W_K: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]', b_K: 'SizedArr[_coconut.typing.Tuple[N_model,]]', W_V: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]', b_V: 'SizedArr[_coconut.typing.Tuple[N_model,]]', W_O: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]', b_O: 'SizedArr[_coconut.typing.Tuple[N_model,]]', residual: 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]',) -> 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]':  #221 (line in Coconut source)

    Q: 'SizedArr[_coconut.typing.Tuple[N_seq, N_heads, N_head_size]]' = (attn_prep)(W_Q, b_Q, residual)  #233 (line in Coconut source)
    K: 'SizedArr[_coconut.typing.Tuple[N_seq, N_heads, N_head_size]]' = (attn_prep)(W_K, b_K, residual)  #234 (line in Coconut source)
    V: 'SizedArr[_coconut.typing.Tuple[N_seq, N_heads, N_head_size]]' = (attn_prep)(W_V, b_V, residual)  #235 (line in Coconut source)

    attn: 'SizedArr[_coconut.typing.Tuple[N_seq, N_seq, N_heads]]' = einsum("attn[q,k,h] = Q[q,h,d] * K[k,h,d]", Q, K)  #237 (line in Coconut source)
    attn /= n_head_size**0.5  #238 (line in Coconut source)
    attn += np.where(mask, 0, float("-inf"))  #239 (line in Coconut source)
    attn = (softmax)(attn, axis=1)  #240 (line in Coconut source)

    output: 'SizedArr[_coconut.typing.Tuple[N_seq, N_heads, N_head_size]]' = einsum("residual[q,h,d] = attn[q,k,h] * V[k,h,d]", attn, V)  #242 (line in Coconut source)
    new_residual: 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]' = output.reshape(n_seq, n_model)  #243 (line in Coconut source)
    return (new_residual @ W_O + b_O)  #244 (line in Coconut source)


def feed_forward(W_ff_in: 'SizedArr[_coconut.typing.Tuple[N_model, N_ff]]', b_ff_in: 'SizedArr[_coconut.typing.Tuple[N_ff,]]', W_ff_out: 'SizedArr[_coconut.typing.Tuple[N_ff, N_model]]', b_ff_out: 'SizedArr[_coconut.typing.Tuple[N_model,]]', residual: 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]',) -> 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]':  #246 (line in Coconut source)
    ff: 'SizedArr[_coconut.typing.Tuple[N_seq, N_ff]]' = residual @ W_ff_in + b_ff_in  #253 (line in Coconut source)
    ff = (activation)(ff)  #254 (line in Coconut source)
    return (ff @ W_ff_out + b_ff_out)  #255 (line in Coconut source)


def activation(ff: 'SizedArr[_coconut.typing.Tuple[N_seq, N_ff]]',) -> 'SizedArr[_coconut.typing.Tuple[N_seq, N_ff]]':  #257 (line in Coconut source)
    return (ff * norm.cdf(ff))  #260 (line in Coconut source)


def transformer_layer(ln_gain_attn: 'SizedArr[_coconut.typing.Tuple[N_model,]]', ln_bias_attn: 'SizedArr[_coconut.typing.Tuple[N_model,]]', W_Q: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]', b_Q: 'SizedArr[_coconut.typing.Tuple[N_model,]]', W_K: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]', b_K: 'SizedArr[_coconut.typing.Tuple[N_model,]]', W_V: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]', b_V: 'SizedArr[_coconut.typing.Tuple[N_model,]]', W_O: 'SizedArr[_coconut.typing.Tuple[N_model, N_model]]', b_O: 'SizedArr[_coconut.typing.Tuple[N_model,]]', ln_gain_ff: 'SizedArr[_coconut.typing.Tuple[N_model,]]', ln_bias_ff: 'SizedArr[_coconut.typing.Tuple[N_model,]]', W_ff_in: 'SizedArr[_coconut.typing.Tuple[N_model, N_ff]]', b_ff_in: 'SizedArr[_coconut.typing.Tuple[N_ff,]]', W_ff_out: 'SizedArr[_coconut.typing.Tuple[N_ff, N_model]]', b_ff_out: 'SizedArr[_coconut.typing.Tuple[N_model,]]', residual: 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]',) -> 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]':  #262 (line in Coconut source)

    residual = (layer_norm)(ln_gain_attn, ln_bias_attn, residual)  #282 (line in Coconut source)

    attn: 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]' = (multi_head_attn)(W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O, residual)  #284 (line in Coconut source)
    residual += attn  #285 (line in Coconut source)

    residual = (layer_norm)(ln_gain_ff, ln_bias_ff, residual)  #287 (line in Coconut source)

    ff: 'SizedArr[_coconut.typing.Tuple[N_seq, N_model]]' = (feed_forward)(W_ff_in, b_ff_in, W_ff_out, b_ff_out, residual)  #289 (line in Coconut source)
    residual += ff  #290 (line in Coconut source)

    return (residual)  #292 (line in Coconut source)


one_layer_transformer = _coconut.functools.partial(transformer_model, [_coconut.functools.partial(transformer_layer, ln_gain_attn, ln_bias_attn, W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O, ln_gain_ff, ln_bias_ff, W_ff_in, b_ff_in, W_ff_out, b_ff_out),])  #294 (line in Coconut source)
