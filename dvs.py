import numpy as np

def generate_events(L, Lref, C, t):
    # Создаёт события кадра и изменяет опору Lref на месте.
    if not np.isfinite(C) or C <= 0:
        raise ValueError("порог контраста C должен быть конечным положительным числом")
    
    if not np.isfinite(t) or t < 0:
        raise ValueError("время должно быть конечным неотрицательным числом")
    
    if not isinstance(Lref, np.ndarray) or not np.issubdtype(Lref.dtype, np.floating):
        raise ValueError("Lref должен быть массивом numpy с вещественным типом")
    L = np.asarray(L)
    
    if L.ndim != 2 or L.shape != Lref.shape or not L.size:
        raise ValueError("L и Lref должны быть непустыми двумерными массивами одинаковой формы")
    
    if not np.isfinite(L).all() or not np.isfinite(Lref).all():
        raise ValueError("логарифмическая яркость должна содержать только конечные значения")
    
    if not Lref.flags.writeable:
        raise ValueError("массив Lref должен быть доступен для изменения")
    
    events = []
    
    diff_log = L - Lref
    on_mask = diff_log >= C
    off_mask = diff_log <= -C
    
    while (np.any(on_mask) or np.any(off_mask)):
        # Учитываем округление при записи в тип Lref, чтобы цикл мог завершиться.
        next_on = (Lref[on_mask] + C).astype(Lref.dtype, copy=False)
        next_off = (Lref[off_mask] - C).astype(Lref.dtype, copy=False)
        if (np.any(next_on == Lref[on_mask])
                or np.any(next_off == Lref[off_mask])):
            raise ValueError("порог C слишком мал для точности типа Lref")
        on_y, on_x = np.nonzero(on_mask)
        off_y, off_x = np.nonzero(off_mask)
    
        for i in range(len(on_x)):
            events.append((on_x[i], on_y[i], t, 1))
        Lref[on_mask] += C
    
        for i in range(len(off_x)):
            events.append((off_x[i], off_y[i], t, -1))
        Lref[off_mask] -= C
    
        diff_log = L - Lref
        on_mask = diff_log >= C
        off_mask = diff_log <= -C
                    
    return events
