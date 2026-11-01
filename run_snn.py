import argparse
import json
import math
from pathlib import Path

import torch

from lif import LIFCell
from snn_input import iter_count_frames, load_metadata


def run_lif(run_dir, *, gain=0.25, tau_m_s=0.05, threshold=1.0, device="cpu"):
    if not math.isfinite(gain) or gain < 0:
        raise ValueError("коэффициент gain должен быть конечным неотрицательным числом")
    
    metadata = load_metadata(run_dir)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if device not in ("cpu", "cuda"):
        raise ValueError("параметр device должен иметь значение cpu, cuda или auto")
    
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна; выберите cpu или auto")
    
    neuron = LIFCell(dt_s=1 / metadata["fps"], tau_m_s=tau_m_s, threshold=threshold).to(device)
    membrane = None
    steps = []
    consumed_events = 0
    
    with torch.no_grad():
        
        for frame_index, counts in iter_count_frames(run_dir):
            input_events = int(counts.sum(dtype="float64"))
            consumed_events += input_events
            x = torch.from_numpy(counts).unsqueeze(0).to(device)
            spikes, membrane = neuron(gain * x, membrane)
            steps.append({"frame_index": frame_index, "t": frame_index / metadata["fps"], "part_index": frame_index // metadata["chunk_frames"], "input_events": input_events, "output_spikes": int(spikes.sum().item())})
            
    if consumed_events != metadata["total_events"]:
        raise ValueError("число загруженных событий не совпадает с указанным в manifest.json")
    
    return {"run_dir": str(Path(run_dir).resolve()), "device": device,
            "processed_frames": len(steps), "consumed_events": consumed_events,
            "total_spikes": sum(step["output_spikes"] for step in steps),
            "part_reads": metadata["part_count"], "gain": gain, "tau_m_s": tau_m_s,
            "threshold": threshold, "dt_s": 1 / metadata["fps"], "steps": steps}


def main(argv=None):
    parser = argparse.ArgumentParser(description='прямой проход LIF по сохранённым событиям без обучения.')
    parser.add_argument("--run", required=True, type=Path, help="каталог с файлом manifest.json")
    parser.add_argument("--gain", type=float, default=0.25, help="множитель входных событий")
    parser.add_argument("--tau", type=float, default=0.05, help="постоянная времени мембраны в секундах")
    parser.add_argument("--threshold", type=float, default=1., help="порог спайка LIF")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="устройство для LIF")
    parser.add_argument("--stats", type=Path, help="путь к статистике JSON; существующий файл не перезаписывается")
    args = parser.parse_args(argv)
    
    try:
        if args.stats is not None and args.stats.exists():
            raise FileExistsError(f"файл статистики уже существует: {args.stats}")
        result = run_lif(args.run, gain=args.gain, tau_m_s=args.tau, threshold=args.threshold, device=args.device)
        
        if args.stats is not None:
            args.stats.parent.mkdir(parents=True, exist_ok=True)
            with open(args.stats, "x", encoding="utf-8") as file:
                json.dump(result, file, indent=2)
                
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"ошибка: {error}\n")
        
    print(f"кадров: {result['processed_frames']}; входных событий: {result['consumed_events']}; выходных спайков: {result['total_spikes']}")
    print(f"прочитано частей при проходе LIF: {result['part_reads']}; устройство: {result['device']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
