import numpy as np

def events_to_image(frame_events, height, width):
    event_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for event in frame_events:
        
        x, y, t, polarity = event
        
        x = int(x)
        y = int(y)
        
        if polarity == 1:
            event_image[y, x] = (0, 255, 0)
            
        elif polarity == -1:
            event_image[y, x] = (0, 0, 255)
    
    
    return event_image
    
    