#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xeeb0cffd

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

import os  #1 (line in Coconut source)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  #3 (line in Coconut source)

raw_css_file = os.path.join(root_dir, "pygments.css")  #5 (line in Coconut source)
raw_highlight_file = os.path.join(root_dir, "raw_highlight.html")  #6 (line in Coconut source)
full_html_file = os.path.join(root_dir, "point_free_transformer.html")  #7 (line in Coconut source)

def highlight() -> 'None':  #9 (line in Coconut source)
    with open(raw_css_file, "r") as f:  #10 (line in Coconut source)
        raw_css = f.read()  #11 (line in Coconut source)

    with open(raw_highlight_file, "r") as f:  #13 (line in Coconut source)
        raw_highlight = f.read()  #14 (line in Coconut source)

    full_html = """
<!DOCTYPE html>
<html>
<head>
<style>
{raw_css}
</style>
</head>
{raw_highlight}
</html>
    """.strip().format(raw_css=raw_css, raw_highlight=raw_highlight)  #26 (line in Coconut source)

    with open(full_html_file, "w") as f:  #31 (line in Coconut source)
        f.write(full_html)  #32 (line in Coconut source)


if __name__ == "__main__":  #34 (line in Coconut source)
    highlight()  #35 (line in Coconut source)
