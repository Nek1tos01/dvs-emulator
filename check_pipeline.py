import argparse
from pathlib import Path
import unittest


def main(argv=None):
    parser = argparse.ArgumentParser(description='проверки проекта и дополнительный проход LIF по своей записи.')
    parser.add_argument("--run", type=Path, help="сохранённая запись для дополнительной проверки через LIF")
    args = parser.parse_args(argv)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    if not result.wasSuccessful():
        return 1
    
    if args.run is not None:
        from run_snn import run_lif
        stats = run_lif(args.run)
        print(f"запись проверена: кадров {stats['processed_frames']}, событий {stats['consumed_events']}, частей {stats['part_reads']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
