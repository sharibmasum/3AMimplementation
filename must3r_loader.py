import importlib
import inspect
import os
from typing import Any, Callable, Optional


def _call_with_supported_kwargs(fn: Callable[..., Any], **kwargs) -> Any:
    sig = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def load_must3r_naver(
    checkpoint_path: str,
    image_size: int = 512,
    device=None,
) -> Any:
    """
    Concrete loader for https://github.com/naver/must3r.

    This attempts to import the MUSt3R model class from common module paths,
    instantiate it, and load weights from a checkpoint file.
    """
    from must3r.model import load_model

    encoder, decoder = load_model(
        chkpt_path=checkpoint_path,
        device=device or "cpu",
        img_size=image_size,
        verbose=False,
    )
    return {
        "encoder": encoder,
        "decoder": decoder,
        "image_size": image_size,
    }


def load_must3r_from_env(device=None) -> Optional[Any]:
    """
    Load a MUSt3R model using a user-provided loader entrypoint.

    Env vars:
      MUST3R_LOADER="your.module:build_fn"
      MUST3R_CHECKPOINT="/abs/path/to/weights.pth" (optional)
      MUST3R_CONFIG="/abs/path/to/config.yaml" (optional)
    """
    entrypoint = os.getenv("MUST3R_LOADER")
    if not entrypoint:
        return None
    if ":" not in entrypoint:
        raise ValueError("MUST3R_LOADER must be in the form 'module:callable'.")
    module_name, fn_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise AttributeError(f"Callable '{fn_name}' not found in module '{module_name}'.")
    checkpoint = os.getenv("MUST3R_CHECKPOINT")
    config = os.getenv("MUST3R_CONFIG")
    model = _call_with_supported_kwargs(
        fn,
        checkpoint=checkpoint,
        ckpt_path=checkpoint,
        weights=checkpoint,
        config=config,
        device=device,
    )
    if hasattr(model, "to") and device is not None:
        model = model.to(device)
    return model

