import os
import re
import textwrap
from typing import Hashable, List, Tuple, Sequence

import pygments
import pygments.lexers
import pygments.formatters
import torch

# Assumes the optional setup was followed. Best effort otherwise.
normalize_sep = lambda s: os.sep.join(s.split("/"))
HOME = os.environ.get("HOME") + os.sep
ENV_PATH = normalize_sep(".venv/torchscript_ir/lib/python3.8/site-packages/")
CACHE_PATH = normalize_sep(".cache/")


def highlight(code):
  # Useful for debugging.
  return pygments.highlight(str(code), pygments.lexers.PythonLexer(),
                            pygments.formatters.TerminalFormatter(bg='dark'))


def clean_references(graph: str) -> str:
  graph = graph.replace(HOME, "")
  graph = graph.replace(ENV_PATH, "")
  graph = graph.replace(CACHE_PATH, "")
  return graph


def demangle(graph: str) -> str:
  # Used to filter unique submodule graphs.
  return re.sub(".___torch_mangle_[0-9]+", "", graph)


def _get_module_name(module: torch.nn.Module) -> str:
  if hasattr(module, 'original_name'):
    return module.original_name
  else:
    return type(module).__name__


def _get_cleaned_method_graphs(module: torch.nn.Module) -> List[Tuple[str]]:
  methods_and_graphs = []

  for attr in sorted(set(dir(module)) - set(["__class__"])):
    try:
      value = getattr(module, attr)
      if hasattr(value, 'graph'):
        graph = demangle(clean_references(str(value.graph)))
        methods_and_graphs.append((attr, graph))
    except:
      continue

  if not len(methods_and_graphs):
    print("WARNING: encountered module without a 'graph' attr.")
    assert not hasattr(module, "graph_for_types")
    return [("*", "module had no methods with graph attrs.\n")]
  else:
    return methods_and_graphs


def _get_module_graphs(module: str) -> List[Tuple[str]]:
  name = _get_module_name(module)
  methods_and_graphs = _get_cleaned_method_graphs(module)
  graphs = [(name, method, graph) for method, graph in methods_and_graphs]
  return graphs


def get_submodule_graphs(module: str) -> List[Tuple[str]]:
  graphs = []
  graphs.extend(_get_module_graphs(module))
  for submodule in module.children():
    graphs.extend(get_submodule_graphs(submodule))
  return graphs


def filter_unique(values: Sequence[Hashable]) -> List[Hashable]:
  seen = set()
  unique_values = []
  for value in values:
    if value not in seen:
      unique_values.append(value)
      seen.add(value)
  return unique_values


def group_by_graph(graphs):
  graphs_to_names_and_methods = {}
  for name, method, graph in graphs:
    if graph not in graphs_to_names_and_methods:
      graphs_to_names_and_methods[graph] = []
    graphs_to_names_and_methods[graph].append((name, method))

  grouped_graphs = []
  for graph, names_and_methods in graphs_to_names_and_methods.items():
    names_and_methods = [
        f"{name}.{method}" for name, method in names_and_methods
    ]
    names_and_methods = filter_unique(names_and_methods)
    graph = textwrap.indent(graph, '  ')
    grouped_graphs.append("\n".join(names_and_methods + [graph]))
  return grouped_graphs
