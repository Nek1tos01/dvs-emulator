import argparse
import json

from read_mp4 import SAMPLE_VIDEO, convert_video
from run_snn import run_lif


def run_demo(*, max_frames=30, max_side=320, device="cpu", preview=False):
    run_dir = convert_video(SAMPLE_VIDEO, max_frames=max_frames, max_side=max_side, preview=preview)
    result = run_lif(run_dir, device=device)
    
    with open(run_dir / "lif_summary.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        
    return run_dir, result


def main(argv=None):
    parser = argparse.ArgumentParser(description='встроенный пример: видео → события → LIF.')
    parser.add_argument("--max-frames", type=int, default=30, help="Число кадров примера")
    parser.add_argument("--max-side", type=int, default=320, help="Максимальная сторона кадра в пикселях")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="Устройство для LIF")
    parser.add_argument("--preview", action="store_true", help="Показать видео и события в окне")
    args = parser.parse_args(argv)
    
    try:
        directory, result = run_demo(max_frames=args.max_frames, max_side=args.max_side, device=args.device, preview=args.preview)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(f"результаты: {directory}")
    print(f"кадров: {result['processed_frames']}; входных событий: {result['consumed_events']}; выходных спайков: {result['total_spikes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
