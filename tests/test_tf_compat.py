import importlib
import sys
import types
import unittest


class TfCompatTest(unittest.TestCase):
    def test_keeps_default_adam_when_legacy_adam_constructor_is_unsupported(self):
        class DefaultAdam:
            pass

        class UnsupportedLegacyAdam:
            def __init__(self, *args, **kwargs):
                raise ImportError("legacy unsupported")

        optimizers = types.SimpleNamespace(
            Adam=DefaultAdam,
            legacy=types.SimpleNamespace(Adam=UnsupportedLegacyAdam),
        )
        fake_tf = types.SimpleNamespace(keras=types.SimpleNamespace(optimizers=optimizers))

        original_tensorflow = sys.modules.get("tensorflow")
        original_tf_compat = sys.modules.pop("tf_compat", None)
        sys.modules["tensorflow"] = fake_tf
        try:
            importlib.import_module("tf_compat")
            self.assertIs(DefaultAdam, fake_tf.keras.optimizers.Adam)
        finally:
            sys.modules.pop("tf_compat", None)
            if original_tf_compat is not None:
                sys.modules["tf_compat"] = original_tf_compat
            if original_tensorflow is None:
                sys.modules.pop("tensorflow", None)
            else:
                sys.modules["tensorflow"] = original_tensorflow


if __name__ == "__main__":
    unittest.main()
