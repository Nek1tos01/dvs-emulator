import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from dvs import generate_events
from events_io import save_events
from visualization import events_to_image

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_VIDEO = PROJECT_DIR / "Videos" / "IMG_7181.MP4"


def convert_video(input_path, output_dir=None, *, threshold=0.2, chunk_frames=5, max_frames=None, max_side=None, preview=False):
    
    # первый кадр задаёт опору, время событий равно индексу кадра / FPS
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("порог threshold должен быть конечным положительным числом")
    
    for name, value in (("chunk_frames", chunk_frames), ("max_frames", max_frames), ("max_side", max_side)):
        if value is None and name != "chunk_frames":
            continue
        
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} должен быть целым положительным числом")
        
    threshold = float(threshold)
    chunk_frames = int(chunk_frames)
    max_frames = None if max_frames is None else int(max_frames)
    max_side = None if max_side is None else int(max_side)

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"видеофайл не найден: {source}")
    
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = PROJECT_DIR / "outputs" / f"{source.stem}_{stamp}"
        
    output = Path(output_dir).expanduser().resolve()
    
    if output.exists():
        raise FileExistsError(f"выберите новый каталог для результатов: {output}")

    cap = cv2.VideoCapture(str(source))
    counter, total_events = 0, 0
    events, parts = [], []

    def save_part():
        nonlocal events, total_events
        name = f"events_part_{len(parts)}.npy"
        total_events += save_events(events, output / name)
        parts.append(name)
        events = []

    try:
        if not cap.isOpened():
            raise ValueError(f"не удалось открыть видео: {source}")
        
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("не удалось получить конечное положительное значение FPS видео")
        
        reported_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        reported_count = int(reported_count) if np.isfinite(reported_count) and reported_count > 0 else None
        ret, frame = cap.read()
        
        if not ret or frame is None:
            raise ValueError("не удалось декодировать ни одного кадра видео")
        
        original_height, original_width = frame.shape[:2]
        output.mkdir(parents=True, exist_ok=False)
        stop_reason = "end_of_stream"

        while True:
            if max_side is not None and max(frame.shape[:2]) > max_side:
                scale = max_side / max(frame.shape[:2])
                size = (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale)))
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            if counter and gray.shape != Lref.shape:
                raise ValueError("размер кадров видео изменился во время декодирования")
            
            safe_frame = np.maximum(gray, 1).astype(np.float32)
            L = np.log(safe_frame)
            t = counter / fps
            frame_events = []
            
            if counter == 0:
                Lref = L.copy()
            else:
                frame_events = generate_events(L, Lref, threshold, t)
                events.extend(frame_events)
            counter += 1
            
            if counter % chunk_frames == 0:
                save_part()

            if preview:
                image = events_to_image(frame_events, height, width)
                cv2.imshow("Видео | ON зелёные, OFF красные", np.hstack((frame, image)))
                
                if cv2.waitKey(max(1, round(1000 / fps))) == ord("q"):
                    stop_reason = "user_stop"
                    break
                
            if max_frames is not None and counter >= max_frames:
                stop_reason = "frame_limit"
                break
            
            ret, frame = cap.read()
            
            if not ret or frame is None:
                
                if reported_count is not None and counter < reported_count:
                    stop_reason = "read_incomplete"
                break

        if counter % chunk_frames != 0:
            save_part()
        metadata = {
            "schema_version": 1, "source_video": str(source),
            "original_height": original_height, "original_width": original_width,
            "height": height, "width": width, "fps": fps,
            "contrast_threshold": threshold, "intensity_floor": 1,
            "time_unit": "seconds", "timestamp_model": "frame_index / fps",
            "columns": ["x", "y", "t", "polarity"],
            "polarity": {"ON": 1, "OFF": -1},
            "processed_frames": counter, "reported_frame_count": reported_count,
            "part_count": len(parts), "parts": parts, "chunk_frames": chunk_frames,
            "event_dtype": "float64", "total_events": total_events,
            "stop_reason": stop_reason,
        }
        
        temporary_manifest = output / "manifest.json.tmp"
        
        with open(temporary_manifest, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
            
        temporary_manifest.replace(output / "manifest.json")
        
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='преобразование видео в события DVS с сохранением по частям.')
    parser.add_argument("--input", type=Path, default=SAMPLE_VIDEO, help="входной MP4/MOV; по умолчанию встроенный пример")
    parser.add_argument("--output", type=Path, help="новый каталог результатов; по умолчанию outputs/<видео>_<дата_и_время>")
    parser.add_argument("--threshold", type=float, default=0.2, help="порог логарифмического контраста")
    parser.add_argument("--chunk-frames", type=int, default=5, help="число кадров в одной сохраняемой части")
    parser.add_argument("--max-frames", type=int, help="лимит кадров; по умолчанию всё видео")
    parser.add_argument("--max-side", type=int, help="максимальная сторона кадра с сохранением пропорций")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--preview", action="store_true", help="окно предпросмотра; q завершает обработку")
    display.add_argument("--no-preview", action="store_false", dest="preview", help="обработка без окна предпросмотра")
    parser.set_defaults(preview=False)
    args = parser.parse_args(argv)
    
    try:
        result = convert_video(args.input, args.output, threshold=args.threshold, chunk_frames=args.chunk_frames, max_frames=args.max_frames, max_side=args.max_side, preview=args.preview)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"ошибка: {error}\n")
        
    print(f"запись сохранена: {result}")
    
    with open(result / "manifest.json", encoding="utf-8") as file:
        metadata = json.load(file)
        
    stop_reasons = {"frame_limit": "достигнут лимит кадров", "user_stop": "остановлено пользователем", "end_of_stream": "конец потока", "read_incomplete": "чтение прервано раньше ожидаемого"}
    
    print(f"кадров: {metadata['processed_frames']}; причина остановки: {stop_reasons[metadata['stop_reason']]}")
    
    if metadata["stop_reason"] == "read_incomplete":
        print("предупреждение: чтение остановилось раньше заявленного числа кадров; проверьте запись или декодер.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
