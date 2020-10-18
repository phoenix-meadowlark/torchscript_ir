import argparse
import os

import utils
import torch

VISION_GITHUB = "pytorch/vision"
ALL_MODELS = torch.hub.list(VISION_GITHUB)


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--models",
      default="all",
      help=f"Either 'all' or a comma delimited list of {ALL_MODELS}.")
  parser.add_argument("--save_models",
                      action="store_true",
                      help="Whether or not to save ")
  return parser.parse_args()


def main():
  args = parse_arguments()
  models = ALL_MODELS if args.models == "all" else args.models.split(",")

  for i, model_name in enumerate(models):
    print(f"Compiling {i + 1:2d}/{len(models)}: {model_name}")
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

    if args.save_models:
      torch.jit.save(script_model,
                     os.path.join(model_path, f"{model_name}_script.pt"))


if __name__ == "__main__":
  main()
