import numpy as np

def validate_events(events, height=None, width=None):
    array = np.asarray(events, dtype=np.float64)
    if array.shape == (0,):
        array = array.reshape(0, 4)
        
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("массив событий должен иметь форму (N, 4)")
    
    if not np.isfinite(array).all():
        raise ValueError("массив событий должен содержать только конечные значения")
    
    if np.any(array[:, :2] < 0) or np.any(array[:, :2] != np.floor(array[:, :2])):
        raise ValueError("координаты пикселей должны быть целыми неотрицательными числами")
    
    if np.any(array[:, 2] < 0):
        raise ValueError("время события в секундах не должно быть отрицательным")
    
    if not np.isin(array[:, 3], [-1, 1]).all():
        raise ValueError("полярность должна быть +1 (ON) или -1 (OFF)")
    
    if height is not None and np.any(array[:, 1] >= height):
        raise ValueError("координата y события выходит за границы изображения")
    
    if width is not None and np.any(array[:, 0] >= width):
        raise ValueError("координата x события выходит за границы изображения")
    
    return array


def save_events(events, filename):
    events_array = validate_events(events)
    
    np.save(filename, events_array)
    
    return len(events_array)
