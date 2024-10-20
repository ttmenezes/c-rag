import torch
from transformers import AutoConfig
import numpy as np
import os

# Load the model weights from the local file
model_path = "pytorch_model.bin"
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location="cpu")
else:
    raise FileNotFoundError(f"The file {model_path} does not exist.")

# Load the model configuration
config_path = "config.json"
if os.path.exists(config_path):
    config = AutoConfig.from_pretrained(config_path)
else:
    raise FileNotFoundError(f"The file {config_path} does not exist.")

# Function to get weights from a state dict
def get_weights(prefix):
    weight = state_dict[f"{prefix}.weight"].numpy()
    bias = state_dict[f"{prefix}.bias"].numpy()
    return weight.T, bias

# Extract weights and biases
weights = {
    "token_embeddings": state_dict["embeddings.word_embeddings.weight"].numpy(),
    "position_embeddings": state_dict["embeddings.position_embeddings.weight"].numpy(),
    "token_type_embeddings": state_dict["embeddings.token_type_embeddings.weight"].numpy(),
    "embeddings_layer_norm_weight": state_dict["embeddings.LayerNorm.weight"].numpy(),
    "embeddings_layer_norm_bias": state_dict["embeddings.LayerNorm.bias"].numpy(),
}

for i in range(config.num_hidden_layers):
    weights[f"layer_{i}"] = {
        "attention": {
            "query": get_weights(f"encoder.layer.{i}.attention.self.query"),
            "key": get_weights(f"encoder.layer.{i}.attention.self.key"),
            "value": get_weights(f"encoder.layer.{i}.attention.self.value"),
            "output": get_weights(f"encoder.layer.{i}.attention.output.dense"),
        },
        "attention_layer_norm_weight": state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"].numpy(),
        "attention_layer_norm_bias": state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].numpy(),
        "ffn": {
            "intermediate": get_weights(f"encoder.layer.{i}.intermediate.dense"),
            "output": get_weights(f"encoder.layer.{i}.output.dense"),
        },
        "ffn_layer_norm_weight": state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy(),
        "ffn_layer_norm_bias": state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy(),
    }

weights["pooler"] = get_weights("pooler.dense")

# Save the weights in binary format
with open('model.bin', 'wb') as f:
    # Embeddings
    weights["token_embeddings"].tofile(f)
    weights["position_embeddings"].tofile(f)
    weights["token_type_embeddings"].tofile(f)
    weights["embeddings_layer_norm_weight"].tofile(f)
    weights["embeddings_layer_norm_bias"].tofile(f)

    # Layers
    for i in range(config.num_hidden_layers):
        layer = weights[f"layer_{i}"]
        layer["attention"]["query"][0].tofile(f)
        layer["attention"]["query"][1].tofile(f)
        layer["attention"]["key"][0].tofile(f)
        layer["attention"]["key"][1].tofile(f)
        layer["attention"]["value"][0].tofile(f)
        layer["attention"]["value"][1].tofile(f)
        layer["attention"]["output"][0].tofile(f)
        layer["attention"]["output"][1].tofile(f)
        layer["attention_layer_norm_weight"].tofile(f)
        layer["attention_layer_norm_bias"].tofile(f)
        layer["ffn"]["intermediate"][0].tofile(f)
        layer["ffn"]["intermediate"][1].tofile(f)
        layer["ffn"]["output"][0].tofile(f)
        layer["ffn"]["output"][1].tofile(f)
        layer["ffn_layer_norm_weight"].tofile(f)
        layer["ffn_layer_norm_bias"].tofile(f)

    # Pooler
    weights["pooler"][0].tofile(f)
    weights["pooler"][1].tofile(f)

print(f"Model saved to model.bin")

# Print some information about the model
print(f"Embedding dimension: {config.hidden_size}")
print(f"Number of hidden layers: {config.num_hidden_layers}")
print(f"Number of attention heads: {config.num_attention_heads}")
print(f"Intermediate size: {config.intermediate_size}")
print(f"Maximum sequence length: {config.max_position_embeddings}")
print(f"Vocabulary size: {config.vocab_size}")
