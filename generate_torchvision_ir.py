import os

import utils

import torch

VISION_GITHUB = "pytorch/vision"
ALL_MODELS = torch.hub.list(VISION_GITHUB)


def main():
  for i, model_name in enumerate(ALL_MODELS):
    print(f"Compiling {i + 1:2d}/{len(ALL_MODELS)}: {model_name}")
    model_path = os.path.join("tsir", "torchvision", model_name)
    os.makedirs(model_path, exist_ok=True)

    model = torch.hub.load(VISION_GITHUB, model_name, verbose=False)
    script_model = torch.jit.script(model)

    with open(os.path.join(model_path, "inlined_graph.tsir"), "w") as f:
      f.write(utils.clean_references(str(script_model.inlined_graph)))

    graphs = utils.get_submodule_graphs(script_model)
    graphs = utils.group_by_graph(graphs)
    lines = str(model).split("\n")
    lines.append("")
    lines.extend(graphs)

    with open(os.path.join(model_path, "submodule_graphs.tsir"), "w") as f:
      f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
  main()
