# TorchScript IR

Utilities for converting PyTorch Hub models to TorchScript IR.
Generates an inlined IR file for each model, as well as a
`submodule.tsir` file containing the graphs for each submodule in the
model. Optionally saves `.pt` files with the scripted/traced models.

```shell
# Optional: env setup.
python3 -m venv "${HOME?}/.venv/torchscript_ir"
source "${HOME?}/.venv/torchscript_ir/bin/activate"

# Install dependencies.
python3 -m pip install -r requirements.txt

# Generate IR for all TorchVision models under `tsir/torchvision/`
python3 generate_torchvision_ir.py

# Save scripted models to disk
python3 generate_torchvision_ir.py --models alexnet,vgg11 --save_models
```

```json
// Included vscode settings:
"python.formatting.provider": "yapf",
// The IR is close enough to python that this is helpful for reading it.
"files.associations": {"*.tsir": "python"},
```
