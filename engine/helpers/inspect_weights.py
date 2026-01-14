import orbax.checkpoint as ocp
import absl.logging
import os

# import json
# from {x} import Gemma TODO - will be using as target for the checkpointer.restores()

absl.logging.set_verbosity(absl.logging.ERROR)

checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "gemma-3-270m")
checkpointer = ocp.StandardCheckpointer()
checkpoint = checkpointer.restore(checkpoint_path)

print(f"There are {len(checkpoint)} keys in the checkpoint\n")

for k, v in checkpoint.items():
    keys = list(v.keys())
    print(f"{k}: {v[keys[0]].shape}")