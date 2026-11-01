import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shutil

import numpy as np
import torch

from dvs import generate_events
from events_io import save_events
from lif import LIFCell
from snn_input import events_to_counts, load_metadata, load_event_part

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "Videos" / "IMG_7181.MP4"


class AlgorithmTests(unittest.TestCase):
    def test_small_double_threshold_does_not_hang_float_reference(self):
        script = """import numpy as np
from dvs import generate_events
try:
    generate_events(np.full((1, 1), 2, dtype=np.float32), np.ones((1, 1), dtype=np.float32), np.float64(1e-9), 0.)
except ValueError:
    print('отклонено')
"""
        try:
            result = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=ROOT,
                                    capture_output=True, text=True, encoding="utf-8", timeout=2)
        except subprocess.TimeoutExpired:
            self.fail("Слишком малый порог должен вызывать ошибку, а не бесконечный цикл")
        self.assertEqual(result.stdout.strip(), "отклонено")

    def test_static_and_contrast_events(self):
        ref = np.log(np.full((2, 2), 50, dtype=np.float32))
        self.assertEqual(generate_events(ref.copy(), ref, .2, .01), [])
        current = np.log(np.full((2, 2), 75, dtype=np.float32))
        events = generate_events(current, ref, .2, .01)
        self.assertEqual(len(events), 8)
        self.assertTrue(all(event[3] == 1 for event in events))
        self.assertTrue(np.all(np.abs(current - ref) < .2))

    def test_reject_nan_threshold(self):
        ref = np.zeros((1, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            generate_events(ref.copy(), ref, float("nan"), 0.)

    def test_counts_preserve_duplicates_and_channels(self):
        counts = events_to_counts([(1, 0, .1, 1), (1, 0, .1, 1), (1, 0, .1, -1)], 2, 2)
        self.assertEqual(counts.shape, (2, 2, 2))
        self.assertEqual(counts.dtype, np.float32)
        self.assertEqual(counts[0, 0, 1], 2)
        self.assertEqual(counts[1, 0, 1], 1)

    def test_bad_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            events_to_counts([(-1, 0, .1, 1)], 2, 2)

    def test_unknown_polarity_is_rejected(self):
        with self.assertRaises(ValueError):
            events_to_counts([(0, 0, .1, 0)], 2, 2)

    def test_empty_saved_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "empty.npy"
            self.assertEqual(save_events([], path), 0)
            self.assertEqual(np.load(path).shape, (0, 4))

    def test_malformed_records_are_not_reshaped_silently(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                save_events([[0, 1], [2, 3]], Path(folder) / "bad.npy")

    def test_lif_accumulation_and_gradient(self):
        neuron = LIFCell(dt_s=1 / 30)
        state = None
        values = []
        for _ in range(3):
            spikes, state = neuron(torch.tensor([.6]), state)
            values.append(spikes.item())
        self.assertEqual(values, [0., 0., 1.])
        x = torch.tensor([.9], requires_grad=True)
        spikes, _ = neuron(x)
        spikes.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.item(), 0)


class PublicationTests(unittest.TestCase):
    def test_zero_processed_frames_are_not_a_valid_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            info = dict(height=1, width=1, fps=30., time_unit="seconds",
                        columns=["x", "y", "t", "polarity"], processed_frames=0,
                        part_count=0, chunk_frames=5, event_dtype="float64", total_events=0)
            (Path(folder) / "manifest.json").write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_metadata(folder)

    def test_numpy_scalar_options_serialize(self):
        convert = importlib.import_module("read_mp4").convert_video
        with tempfile.TemporaryDirectory() as folder:
            path = convert(SAMPLE, Path(folder) / "run", threshold=np.float32(.2),
                           chunk_frames=np.int64(5), max_frames=np.int64(2), max_side=np.int64(32))
            self.assertEqual(load_metadata(path)["processed_frames"], 2)

    def test_premature_read_stop_is_distinguishable(self):
        from unittest.mock import patch
        module = importlib.import_module("read_mp4")
        class Capture:
            def __init__(self):
                self.reads = 0
                self.released = False
            def isOpened(self):
                return True
            def get(self, prop):
                return 30. if prop == module.cv2.CAP_PROP_FPS else 3.
            def read(self):
                self.reads += 1
                return (True, np.full((4, 4, 3), 50, dtype=np.uint8)) if self.reads == 1 else (False, None)
            def release(self):
                self.released = True
        capture = Capture()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(module.cv2, "VideoCapture", return_value=capture):
                path = module.convert_video(SAMPLE, Path(folder) / "run")
            self.assertEqual(load_metadata(path)["stop_reason"], "read_incomplete")
            self.assertTrue(capture.released)

    def test_package_cannot_overwrite_non_zip_file(self):
        from tools.build_colab_zip import build_zip
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "README.md"
            path.write_text("исходный текст", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_zip(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "исходный текст")

    def test_package_cannot_be_placed_among_its_inputs(self):
        from unittest.mock import patch
        from tools import build_colab_zip
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "source"
            shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(
                ".git", ".venv", "outputs", ".publication-work", "__pycache__", "*.npy"))
            with patch.object(build_colab_zip, "ROOT", root):
                with self.assertRaises(ValueError):
                    build_colab_zip.build_zip(root / "tools" / "bundle.zip")

    def test_cli_works_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            converted = subprocess.run(
                [sys.executable, "-X", "utf8", str(ROOT / "read_mp4.py"), "--output", str(output),
                 "--max-frames", "2", "--max-side", "32", "--no-preview"],
                cwd=folder, capture_output=True, text=True, encoding="utf-8", timeout=40,
            )
            self.assertEqual(converted.returncode, 0, converted.stderr)
            stats = Path(folder) / "stats.json"
            inferred = subprocess.run(
                [sys.executable, "-X", "utf8", str(ROOT / "run_snn.py"), "--run", str(output), "--stats", str(stats)],
                cwd=folder, capture_output=True, text=True, encoding="utf-8", timeout=40,
            )
            self.assertEqual(inferred.returncode, 0, inferred.stderr)
            result = json.loads(stats.read_text(encoding="utf-8"))
            self.assertEqual(result["processed_frames"], 2)

    def test_manifest_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            info = dict(schema_version=999, height=1, width=1, fps=30., time_unit="seconds",
                        columns=["x", "y", "t", "polarity"], processed_frames=1,
                        part_count=1, chunk_frames=5, event_dtype="float64", total_events=0)
            (Path(folder) / "manifest.json").write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_metadata(folder)

    def test_preview_stop_saves_empty_tail(self):
        from unittest.mock import patch
        module = importlib.import_module("read_mp4")
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(module.cv2, "imshow"), patch.object(module.cv2, "waitKey", return_value=ord("q")), patch.object(module.cv2, "destroyAllWindows"):
                output = module.convert_video(SAMPLE, Path(folder) / "run", preview=True, max_side=32)
            info = load_metadata(output)
            self.assertEqual(info["processed_frames"], 1)
            self.assertEqual(info["stop_reason"], "user_stop")
            self.assertEqual(load_event_part(output, 0).shape, (0, 4))

    def test_conversion_module_import_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", "import read_mp4; print(callable(getattr(read_mp4, 'convert_video', None)))"],
                cwd=folder, env=env, capture_output=True, text=True, encoding="utf-8", timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "True")
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_chunked_conversion_and_partial_tail(self):
        convert = importlib.import_module("read_mp4").convert_video
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            result = convert(SAMPLE, output, max_frames=7, max_side=64, chunk_frames=5)
            self.assertEqual(result, output.resolve())
            info = load_metadata(result)
            self.assertEqual(info["processed_frames"], 7)
            self.assertEqual(info["part_count"], 2)
            self.assertEqual(info["stop_reason"], "frame_limit")
            self.assertEqual(max(info["height"], info["width"]), 64)
            parts = [load_event_part(result, i) for i in range(2)]
            self.assertEqual(sum(len(part) for part in parts), info["total_events"])
            self.assertTrue(all(part.shape[1:] == (4,) for part in parts))
            with self.assertRaises(FileExistsError):
                convert(SAMPLE, output, max_frames=1)

    def test_failed_input_creates_no_output(self):
        convert = importlib.import_module("read_mp4").convert_video
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            with self.assertRaises(FileNotFoundError):
                convert(Path(folder) / "missing.mp4", output)
            self.assertFalse(output.exists())

    def test_undecodable_video_creates_no_output(self):
        convert = importlib.import_module("read_mp4").convert_video
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "bad.mp4"
            source.write_bytes(b"not a video")
            output = Path(folder) / "run"
            with self.assertRaises(ValueError):
                convert(source, output)
            self.assertFalse(output.exists())

    def test_snn_state_and_empty_frames_across_parts(self):
        self.assertIsNotNone(importlib.util.find_spec("run_snn"), "Модуль run_snn не найден")
        run_lif = importlib.import_module("run_snn").run_lif
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root / "events_part_0.npy", np.array([[0, 0, 1 / 30, 1]], dtype=float))
            np.save(root / "events_part_1.npy", np.array([[0, 0, 2 / 30, 1], [0, 0, 3 / 30, 1]], dtype=float))
            info = dict(height=1, width=1, fps=30., contrast_threshold=.2, time_unit="seconds",
                        columns=["x", "y", "t", "polarity"], processed_frames=4, part_count=2,
                        chunk_frames=2, event_dtype="float64", total_events=3)
            (root / "manifest.json").write_text(json.dumps(info), encoding="utf-8")
            result = run_lif(root, gain=.6)
            self.assertEqual([s["output_spikes"] for s in result["steps"]], [0, 0, 0, 1])
            self.assertEqual(result["consumed_events"], 3)
            self.assertEqual(result["part_reads"], 2)


if __name__ == "__main__":
    unittest.main()
