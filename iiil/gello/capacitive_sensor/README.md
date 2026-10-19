# Capacitive Touch Sensor (ESP32-S3 + MPR121)

Firmware for the LilyGo T-Display ESP32-S3R8 that reads an Adafruit MPR121
capacitive touch sensor over I2C and sends touch state to the host PC over
USB Serial (CDC).

## Hardware

- **MCU**: LilyGo T-Display ESP32-S3R8
- **Sensor**: Adafruit MPR121 12-Key Capacitive Touch Sensor Breakout
- **Connection**: STEMMA QT cable (JST SH 4-pin, I2C)
- **Touch Input**: Conductive copper tape connected to MPR121 channel 11

## Wiring

Connect the MPR121 breakout to the T-Display S3 via STEMMA QT cable:

| MPR121 Pin | ESP32-S3 Pin | Wire Color |
|------------|-------------|------------|
| VCC        | 3.3V        | Red        |
| GND        | GND         | Black      |
| SCL        | GPIO 44     | Yellow     |
| SDA        | GPIO 43     | Green      |

If not using STEMMA QT, solder directly to the header pins.

Attach conductive copper tape to the electrode pads on the MPR121 breakout
(pads 0-11). For single-channel use, connect tape to pad 11.

## Flashing

### Arduino IDE

1. Install board support: Add ESP32 board URL to Preferences, install "esp32" boards
2. Install libraries: Sketch > Include Library > Manage Libraries, then install:
   - "Adafruit MPR121"
   - "Arduino_GFX_Library" (by moononournation)
3. Select board: Tools > Board > ESP32S3 Dev Module
4. Configure:
   - USB CDC On Boot: **Enabled**
   - Flash Size: 16MB (3MB APP/9.9MB FATFS)
   - PSRAM: OPI PSRAM
5. Select port and upload

### PlatformIO

```ini
[env:t-display-s3]
platform = espressif32
board = lilygo-t-display-s3
framework = arduino
lib_deps =
    adafruit/Adafruit MPR121@^1.1.2
    moononournation/Arduino_GFX_Library
build_flags = -DARDUINO_USB_CDC_ON_BOOT=1
```

## Serial Protocol

Output at 200Hz (every 5ms) over USB CDC at 115200 baud:

| Message | Meaning |
|---------|---------|
| `T000`  | No channels touched |
| `T800`  | Channel 11 touched |
| `T805`  | Channels 0, 2, and 11 touched |
| `TFFF`  | Channels 0-11 all touched |
| `E01`   | MPR121 not found (check wiring) |
| `E02`   | MPR121 communication error |

The hex bitmask represents channels 0-11 (3 hex digits, 12 bits).

## USB Port Setup

Create a udev rule for a stable device symlink:

```bash
sudo cp scripts/99-touch-sensor.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Verify: `ls -la /dev/tty_touch_sensor`

## Tuning

### Touch Sensitivity

In the firmware, adjust `TOUCH_THRESHOLD` and `RELEASE_THRESHOLD`:

- **Lower values** = more sensitive (detects lighter touch)
- **Higher values** = less sensitive (requires firmer contact)
- Touch threshold must be greater than release threshold
- Defaults: touch=12, release=6

### Output Rate

Adjust `SEND_INTERVAL_MS` (default 5ms = 200Hz). The host-side Python reader
works at any rate; the 100Hz control loop reads cached state.

## Troubleshooting

- **E01 errors**: Check I2C wiring, verify MPR121 address (default 0x5A)
- **E02 errors**: I2C communication failure, check cable connection
- **No serial output**: Ensure "USB CDC On Boot: Enabled" in board settings
- **False touches**: Increase TOUCH_THRESHOLD, check for nearby metal interference
- **Missed touches**: Decrease TOUCH_THRESHOLD, ensure good contact with tape
