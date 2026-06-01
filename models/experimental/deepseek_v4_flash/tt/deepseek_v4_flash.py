import ttnn
import torch

from .weight_loader import DeepseekV4WeightLoader


class DeepSeekV4Flash(ttnn.Module):
    """Stub V4-Flash model wired up to the safetensors weight loader.

    Only the embedding table is materialised today; the rest of the model is
    a placeholder. The loader, however, can already serve every parameter in
    the checkpoint (see ``load_state_dict_torch``), so as more submodules are
    fleshed out they can be populated with one ``loader.get_tensor(...)`` per
    parameter.
    """

    def __init__(self, config: dict, weights_dir: str):
        super().__init__()
        self.config = config
        self.weights_dir = weights_dir
        self.weight_loader = DeepseekV4WeightLoader(weights_dir)
        self.embed_tokens = ttnn.Embedding(config["vocab_size"], config["hidden_size"])

    def load_weights(self) -> None:
        """Populate the model's submodules from the safetensors checkpoint.

        Currently fills only ``embed_tokens`` (the rest of the model is a
        stub); extend this as new submodules land.
        """
        embed = self.weight_loader.get_tensor("embed_tokens.weight")
        self.embed_tokens.weight = ttnn.from_torch(embed)

    def load_state_dict_torch(self, hf_names: list[str]) -> dict[str, torch.Tensor]:
        """Return a ``{hf_name: torch.Tensor}`` dict for the given names.

        Convenience for tests / reference comparisons that want the raw
        torch tensors keyed by HF-style names.
        """
        return {name: self.weight_loader.get_tensor(name) for name in hf_names}

    def forward(self, input_ids: ttnn.Tensor, attention_mask: ttnn.Tensor) -> ttnn.Tensor:
        input_embeddings = self.embed_tokens(input_ids)
