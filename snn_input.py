import json
from pathlib import Path

import numpy as np

from events_io import validate_events


def load_metadata(run_dir):
    with open(Path(run_dir) / "manifest.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)
        
    if not isinstance(metadata, dict):
        raise ValueError("файл manifest.json должен содержать объект JSON")
    version = metadata.get("schema_version")
    
    if version is not None and (type(version) is not int or version != 1):
        raise ValueError("версия формата manifest.json не поддерживается")
    
    for key in ("height", "width", "chunk_frames", "processed_frames", "part_count", "total_events"):
        value = metadata.get(key)
        minimum = 1 if key in ("height", "width", "chunk_frames", "processed_frames") else 0
        
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"недопустимое поле в manifest.json: {key}")
        
    fps = metadata.get("fps")
    
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS в manifest.json должен быть конечным положительным числом")
    
    if metadata.get("time_unit") != "seconds" or metadata.get("columns") != ["x", "y", "t", "polarity"]:
        raise ValueError("столбцы событий или единица времени не поддерживаются")
    
    if metadata.get("event_dtype") != "float64":
        raise ValueError("тип данных событий не поддерживается")
    
    expected_parts = (metadata["processed_frames"] + metadata["chunk_frames"] - 1) // metadata["chunk_frames"]
    
    if metadata["part_count"] != expected_parts:
        raise ValueError("число частей не соответствует разбиению кадров")
    
    parts = metadata.get("parts")
    
    if parts is not None:
        
        if (not isinstance(parts, list) or len(parts) != expected_parts
                or any(not isinstance(p, str) or not p or Path(p).name != p or p in (".", "..") for p in parts)
                or len(set(parts)) != len(parts)):
            raise ValueError("список parts должен содержать уникальные имена файлов в порядке времени")
        
    return metadata


def load_event_part(run_dir, part_index):
    metadata = load_metadata(run_dir)
    
    if isinstance(part_index, bool) or not isinstance(part_index, (int, np.integer)) or not 0 <= part_index < metadata["part_count"]:
        raise ValueError("индекс части выходит за границы записи")
    
    parts = metadata.get("parts")
    filename = parts[part_index] if parts is not None else f"events_part_{part_index}.npy"
    events = np.load(Path(run_dir) / filename, allow_pickle=False)
    
    if events.dtype != np.float64:
        raise ValueError("тип данных части не совпадает с указанным в manifest.json")
    
    events = validate_events(events, metadata["height"], metadata["width"])
    frame_ids = np.rint(events[:, 2] * metadata["fps"])
    
    if np.any(np.abs(events[:, 2] * metadata["fps"] - frame_ids) > 1e-5):
        raise ValueError("время событий не соответствует кадровым меткам эмулятора")
    
    if (np.any(frame_ids >= metadata["processed_frames"])
            or np.any(frame_ids // metadata["chunk_frames"] != part_index)
            or np.any(np.diff(events[:, 2]) < 0)):
        raise ValueError("время событий не соответствует разбиению кадров по частям")
    
    return events


def events_to_counts(frame_events, height, width):
    
    for value in (height, width):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError("размеры изображения должны быть целыми положительными числами")
        
    events = validate_events(frame_events, height, width)
    counts = np.zeros((2, height, width), dtype=np.float32)
    
    for x, y, t, polarity in events:
        x, y = int(x), int(y)
        
        if polarity == 1:
            counts[0, y, x] += 1
        elif polarity == -1:
            counts[1, y, x] += 1
            
    return counts


def select_frame_events(events, frame_index, fps):
    
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS должен быть конечным положительным числом")
    
    if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)) or frame_index < 0:
        raise ValueError("индекс кадра должен быть целым неотрицательным числом")
    
    events = validate_events(events)
    frame_ids = np.rint(events[:, 2] * fps)
    
    return events[frame_ids == frame_index]


def iter_count_frames(run_dir):
    #выдаёт все кадры, включая пустые, и хранит в памяти одну часть записи
    metadata = load_metadata(run_dir)
    loaded_part_index = None
    events = None
    for frame_index in range(metadata["processed_frames"]):
        part_index = frame_index // metadata["chunk_frames"]
        if part_index != loaded_part_index:
            events = load_event_part(run_dir, part_index)
            loaded_part_index = part_index
        selected = select_frame_events(events, frame_index, metadata["fps"])
        yield frame_index, events_to_counts(selected, metadata["height"], metadata["width"])
