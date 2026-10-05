import unittest
from unittest.mock import patch
import pandas as pd
from src.improve_root import root_variant, training_variants, ROOT_TRANSFORMS


class RootTests(unittest.TestCase):
    def test_exact_component_changes_and_offline(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("Network access attempted")

        with patch("socket.getaddrinfo", forbidden), patch(
            "requests.sessions.Session.request", forbidden
        ):
            source = "https://user:pass@www.example.org:8443?A=B#Part"
            self.assertEqual(
                root_variant(source, "remove_www_root_slash"),
                "https://user:pass@example.org:8443/?A=B#Part",
            )
            self.assertEqual(
                root_variant("https://www.example.org/Login", "root_slash"),
                "https://www.example.org/Login",
            )
            self.assertEqual(
                root_variant("https://www2.example.org/", "remove_www"),
                "https://www2.example.org/",
            )
            self.assertEqual(
                root_variant("https://[2001:db8::1]:443", "root_slash"),
                "https://[2001:db8::1]:443/",
            )
            for kind in ROOT_TRANSFORMS:
                transformed = root_variant(source, kind)
                self.assertEqual(root_variant(transformed, kind), transformed)

    def test_augmentation_conflicts_duplicates_and_parent_provenance(self):
        train = pd.DataFrame(
            {
                "url": ["https://www.example.org", "https://example.org"],
                "label": [0, 1],
                "source_row": [123, 456],
            }
        )
        augmented, audit = training_variants(train)
        original = augmented[augmented["transform"] == "original"]
        self.assertEqual(original.url.tolist(), train.url.tolist())
        self.assertEqual(original.label.tolist(), train.label.tolist())
        self.assertGreater(audit["conflicting_generated_rows_removed"], 0)
        self.assertEqual(
            len(augmented.drop_duplicates(["url", "label"])), len(augmented)
        )
        self.assertLessEqual(set(augmented.source_row), {123, 456})
        self.assertTrue((augmented.groupby("url").label.nunique() == 1).all())
        same_label = train.copy()
        same_label["label"] = 0
        augmented, _ = training_variants(same_label)
        self.assertEqual(
            set(augmented.loc[augmented["transform"].eq("original"), "source_row"]),
            {123, 456},
        )


if __name__ == "__main__":
    unittest.main()
