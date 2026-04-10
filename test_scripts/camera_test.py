import gi
gi.require_version('Aravis', '0.8')
from gi.repository import Aravis
import numpy as np
import cv2
import os
import time

os.makedirs("frames", exist_ok=True)

CAMERAS = {
    "192.168.0.142": "BFS_23595723",
    "192.168.0.116": "FRAMOS_D71",
    "192.168.1.138": "BFS_23595719",
    "192.168.1.139": "BFS_23595720",
    "192.168.1.143": "BFS_23595724",
    "192.168.1.102": "FRAMOS_D63",
}

cameras = {}
streams = {}

for ip, label in CAMERAS.items():
    try:
        device = Aravis.open_device(ip)
        cam = Aravis.Camera.new_with_device(device)
        
        # Set packet size explicitly to avoid MTU issues
        try:
            cam.gv_set_packet_size(1400)
            print(f"  Packet size set to 1400")
        except Exception as e:
            print(f"  Packet size warning: {e}")

        cam.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
        stream = cam.create_stream(None, None)
        payload = cam.get_payload()
        print(f"  Payload size: {payload} bytes")
        for _ in range(20):
            stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        cam.start_acquisition()
        cameras[label] = cam
        streams[label] = stream
        print(f"Opened {label} @ {ip}")
    except Exception as e:
        print(f"Failed {label} @ {ip}: {e}")

print("\nCapturing 1 frame per camera...")
for label, stream in streams.items():
    cam = cameras[label]  # ← ADD THIS LINE - get the right camera
    w = cam.get_integer('Width')
    h = cam.get_integer('Height')
    pixel_format = cam.get_string('PixelFormat')
    got_frame = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        buf = stream.try_pop_buffer()
        if buf is not None:
            if buf.get_status() == Aravis.BufferStatus.SUCCESS:
                data = buf.get_data()
                pixels = w * h

                if pixel_format == 'Mono16':
                    frame = np.frombuffer(data, dtype=np.uint16).reshape((h, w))
                    frame = (frame >> 8).astype(np.uint8)
                elif pixel_format in ('BayerRG8', 'BayerBG8', 'BayerGB8', 'BayerGR8'):
                    frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
                    bayer_codes = {
                        'BayerRG8': cv2.COLOR_BayerRG2BGR,
                        'BayerBG8': cv2.COLOR_BayerBG2BGR,
                        'BayerGB8': cv2.COLOR_BayerGB2BGR,
                        'BayerGR8': cv2.COLOR_BayerGR2BGR,
                    }
                    frame = cv2.cvtColor(frame, bayer_codes[pixel_format])
                elif pixel_format == 'RGB8':
                    frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                elif pixel_format == 'Mono8':
                    frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
                else:
                    print(f"  Unknown format {pixel_format}, trying Mono8")
                    frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w))

                filename = f"frames/{label}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename} ({w}x{h} {pixel_format})")
                stream.push_buffer(buf)
                got_frame = True
                break
            else:
                stream.push_buffer(buf)
        time.sleep(0.001)
    if not got_frame:
        print(f"No frame from {label}")

for cam in cameras.values():
    cam.stop_acquisition()
# print("Done - check frames/ directory")
