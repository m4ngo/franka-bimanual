import pyrealsense2 as rs

try:
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and enable the FRAMOS camera by IP
    config = rs.config()
    # Replace with your camera's IP address
    config.enable_device_from_file("192.168.0.116")  # For network cameras
    # Or use: config.enable_device("serial_number") for USB

    # Enable streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)
    print("Connected to FRAMOS camera.")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Example: print distance at center pixel
        distance = depth_frame.get_distance(320, 240)
        print(f"Distance at center: {distance:.3f} meters")

except Exception as e:
    print(f"Error: {e}")

finally:
    pipeline.stop()
