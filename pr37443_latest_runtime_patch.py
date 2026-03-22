from pathlib import Path


p = Path("/opt/venv/lib/python3.12/site-packages/vllm/config/speculative.py")
s = p.read_text()

s = s.replace(
    "from collections.abc import Mapping\n",
    "from collections.abc import Callable, Mapping\n",
) if "from collections.abc import Mapping" in s else s

start = s.find("class _DraftHfOverrides")
end = s.find("@config\nclass SpeculativeConfig:")
if start != -1 and end != -1:
    s = s[:start] + """class _DraftHfOverrides:
    \"\"\"Compose target-model HF overrides with draft-model overrides.\"\"\"

    def __init__(
        self,
        target_hf_overrides: Mapping[str, Any]
        | Callable[[PretrainedConfig], PretrainedConfig],
    ) -> None:
        self.target_hf_overrides = target_hf_overrides

    def __call__(self, hf_config: PretrainedConfig) -> PretrainedConfig:
        if isinstance(self.target_hf_overrides, Mapping):
            SpeculativeConfig._apply_hf_overrides_dict(
                hf_config, dict(self.target_hf_overrides)
            )
        else:
            hf_config = self.target_hf_overrides(hf_config)
        return SpeculativeConfig.hf_config_override(hf_config)

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self.target_hf_overrides, Mapping):
            return self.target_hf_overrides.get(key, default)
        return default


""" + s[end:]

old_apply = """    @staticmethod
    def _apply_hf_overrides_dict(
        config: PretrainedConfig,
        overrides: dict[str, Any],
    ) -> None:
        for key, value in overrides.items():
            attr = getattr(config, key, None)
            if isinstance(value, dict) and attr is not None and (
                isinstance(attr, dict) or hasattr(attr, "__dict__")
            ):
                SpeculativeConfig._update_nested_hf_config(attr, value)
            else:
                setattr(config, key, value)
"""
new_apply = """    @staticmethod
    def _apply_hf_overrides_dict(
        config: PretrainedConfig,
        overrides: dict[str, Any],
    ) -> None:
        from transformers import PretrainedConfig

        for key, value in overrides.items():
            attr = getattr(config, key, None)
            if (
                attr is not None
                and isinstance(attr, PretrainedConfig)
                and isinstance(value, dict)
            ):
                SpeculativeConfig._update_nested_hf_config(attr, value)
            else:
                setattr(config, key, value)
"""
s = s.replace(old_apply, new_apply)

old_get = """    @staticmethod
    def _get_draft_hf_overrides(target_hf_overrides: Any) -> Any:
        if isinstance(target_hf_overrides, dict):
            merged_overrides = _DraftHfOverrides(copy.deepcopy(target_hf_overrides))
            return merged_overrides

        if callable(target_hf_overrides):

            def composed_hf_overrides(
                hf_config: PretrainedConfig,
            ) -> PretrainedConfig:
                hf_config = target_hf_overrides(hf_config)
                return SpeculativeConfig.hf_config_override(hf_config)

            return composed_hf_overrides

        return SpeculativeConfig.hf_config_override
"""
new_get = """    @staticmethod
    def _get_draft_hf_overrides(target_hf_overrides: Any) -> Any:
        if isinstance(target_hf_overrides, Mapping):
            if not target_hf_overrides:
                return SpeculativeConfig.hf_config_override
            return _DraftHfOverrides(copy.deepcopy(dict(target_hf_overrides)))

        if callable(target_hf_overrides):
            return _DraftHfOverrides(target_hf_overrides)

        return SpeculativeConfig.hf_config_override
"""
s = s.replace(old_get, new_get)

p.write_text(s)
print("applied PR37443 runtime patch", p)
