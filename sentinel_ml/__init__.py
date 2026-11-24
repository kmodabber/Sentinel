# LightGBM 4.7.0 on Windows segfaults in C-level set_label if pandas is
# imported before lightgbm (numpy 2.0 ABI issue). Pin the order at package
# import so every entry point gets lightgbm loaded first.
import lightgbm as _lightgbm  # noqa: F401
