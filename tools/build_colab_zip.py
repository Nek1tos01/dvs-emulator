import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build_zip(destination):
    destination = Path(destination).resolve()
    if destination.suffix.lower() != ".zip":
        raise ValueError("Путь к архиву должен иметь расширение .zip")
    if any(destination.is_relative_to((ROOT / name).resolve())
           for name in ("tests", "notebooks", "tools", ".github", "Videos")):
        raise ValueError("Сохраните архив вне каталогов исходников, например в outputs/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [ROOT / name for name in (
        ".gitignore", "README.md", "requirements.txt", "requirements-dev.txt",
        "demo.py", "read_mp4.py", "run_snn.py", "dvs.py", "events_io.py",
        "snn_input.py", "lif.py", "visualization.py", "check_pipeline.py", "funcofyark.py",
        "Videos/IMG_7181.MP4",
    )]
    for directory in ("tests", "notebooks", "tools", ".github"):
        files.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and ".ipynb_checkpoints" not in path.parts
                     and path.resolve() != destination)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    return destination


def main():
    parser = argparse.ArgumentParser(description='Архив исходников и примера видео для запуска в Colab.')
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "dvs-emulator-colab.zip", help="Путь к ZIP-архиву")
    args = parser.parse_args()
    print(build_zip(args.output))


if __name__ == "__main__":
    main()
