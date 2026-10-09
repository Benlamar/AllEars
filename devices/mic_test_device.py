
import sounddevice as sd
import numpy as np

devices = sd.query_devices()

print("Speak into your EXTERNAL microphone during each test.")
print("Each device will record for 4 seconds.\n")

for device_id, device in enumerate(devices):
    if device["max_input_channels"] < 1:
        continue

    # Focus on the likely microphone inputs first
    name = device["name"]
    print(f"\nDevice {device_id}: {name}")

    input("Press Enter to record this device...")

    try:
        rate = int(device["default_samplerate"])
        channels = 1

        audio = sd.rec(
            int(4 * rate),
            samplerate=rate,
            channels=channels,
            dtype="float32",
            device=device_id,
        )
        sd.wait()

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        print(f"RMS level:  {rms:.6f}")
        print(f"Peak level: {peak:.6f}")

    except Exception as e:
        print(f"Could not record: {e}")

print("\nDone!")
