"""Check the serialization used by experiment outputs."""

import gzip
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.utils import sha256, write_csv_gzip


class ArtifactIOTests(unittest.TestCase):
    def test_gzip_is_independent_of_time_and_output_filename(self):
        frame = pd.DataFrame(
            {
                "url": ["https://example.org/登入", "https://example.org/?x=1"],
                "label": [0, 1],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv.gz"
            second = Path(directory) / "second.csv.gz"
            write_csv_gzip(frame, first)
            write_csv_gzip(frame, second)
            self.assertEqual(sha256(first), sha256(second))
            header = first.read_bytes()
            self.assertEqual(int.from_bytes(header[4:8], "little"), 0)
            self.assertEqual(header[3] & 8, 0)
            decoded = gzip.decompress(header)
            self.assertNotIn(b"\r\n", decoded)
            pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(decoded)), frame)


if __name__ == "__main__":
    unittest.main()
