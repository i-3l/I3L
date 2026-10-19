/*
 * Capacitive Touch Sensor Firmware
 * LilyGo T-Display ESP32-S3R8 + Adafruit MPR121 (12-key capacitive touch)
 *
 * Reads MPR121 over I2C via STEMMA QT connector and sends touch state
 * over USB Serial (CDC) at 200Hz. Shows status on the built-in LCD.
 *
 * Protocol:
 *   "T<3-hex-digit-bitmask>\n"  - touch data (channels 0-11)
 *   "E<2-hex-digit-code>\n"     - error codes
 *
 * Examples:
 *   "T000\n" = no channels touched
 *   "T800\n" = channel 11 touched
 *   "T805\n" = channels 0, 2, and 11 touched
 *   "E01\n" = MPR121 not found
 *   "E02\n" = MPR121 communication error
 *
 * Hardware:
 *   MPR121 I2C address: 0x5A (default)
 *   I2C pins: GPIO 43 (SDA), GPIO 44 (SCL) via STEMMA QT
 *
 * Dependencies:
 *   - Adafruit MPR121 library (install via Arduino Library Manager)
 *   - Arduino_GFX_Library (install via Arduino Library Manager)
 *   - Wire library (built-in)
 *
 * Board settings (Arduino IDE):
 *   Board: "ESP32S3 Dev Module"
 *   USB CDC On Boot: Enabled
 *   Flash Size: 16MB
 *   PSRAM: OPI PSRAM
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPR121.h>
#include "Arduino_GFX_Library.h"

// ---------- Display Configuration (T-Display S3) ----------

#define GFX_DEV_DEVICE LILYGO_T_DISPLAY_S3
#define GFX_EXTRA_PRE_INIT()              \
    {                                     \
        pinMode(15 /* PWD */, OUTPUT);    \
        digitalWrite(15 /* PWD */, HIGH); \
    }
#define GFX_BL 38

// 170x320 ST7789 via 8-bit parallel bus
Arduino_DataBus *bus = new Arduino_ESP32PAR8Q(
    7 /* DC */, 6 /* CS */, 8 /* WR */, 9 /* RD */,
    39 /* D0 */, 40 /* D1 */, 41 /* D2 */, 42 /* D3 */,
    45 /* D4 */, 46 /* D5 */, 47 /* D6 */, 48 /* D7 */);
Arduino_GFX *gfx = new Arduino_ST7789(
    bus, 5 /* RST */, 0 /* rotation */, true /* IPS */,
    170 /* width */, 320 /* height */,
    35 /* col offset 1 */, 0 /* row offset 1 */,
    35 /* col offset 2 */, 0 /* row offset 2 */);
Arduino_GFX *canvas;

// ---------- Sensor Configuration ----------

#define I2C_SDA 43
#define I2C_SCL 44
#define MPR121_ADDR 0x5A
#define TOUCH_THRESHOLD  195
#define RELEASE_THRESHOLD 117
#define ACTIVE_CHANNEL 11
#define SEND_INTERVAL_MS 5
#define RECONNECT_INTERVAL_MS 2000

// ---------- Display Layout ----------

#define SCREEN_W   170
#define SCREEN_H   320
#define CIRCLE_X   (SCREEN_W / 2)   // 85
#define CIRCLE_Y   160
#define CIRCLE_R   65

// Colors (RGB565)
#define COLOR_BG       0x0000       // Black
#define COLOR_TOUCHED  0x07E0       // Green
#define COLOR_IDLE     0x4208       // Dark gray
#define COLOR_ERROR    0xF800       // Red
#define COLOR_TEXT     0xFFFF       // White
#define COLOR_LABEL    0xBDF7       // Light gray

// ---------- Shared State (mutex-protected) ----------

SemaphoreHandle_t xMutex;
volatile bool     disp_sensorOK  = false;
volatile bool     disp_touched   = false;
volatile uint16_t disp_channels  = 0;
volatile uint16_t disp_filtered  = 0;   // Raw filtered data for active channel
volatile uint16_t disp_baseline  = 0;   // Baseline data for active channel

// ---------- Sensor State (main loop only) ----------

Adafruit_MPR121 cap = Adafruit_MPR121();
bool sensorOK = false;
bool calibrationMode = false;  // 'C' = calibration output, 'N' = normal
unsigned long lastSendTime = 0;
unsigned long lastReconnectTime = 0;
unsigned long lastPingTime = 0;
uint16_t lastTouched = 0;

// How often to verify the sensor is still on the bus (ms)
#define PING_INTERVAL_MS 500

// ---------- MPR121 Register Addresses ----------

#define MPR121_ECR       0x5E  // Electrode Configuration Register
#define MPR121_CONFIG1   0x5C  // CDC config (charge current)
#define MPR121_CONFIG2   0x5D  // CDT config (charge time)
#define MPR121_MHDR      0x2B  // Max Half Delta Rising
#define MPR121_NHDR      0x2C  // Noise Half Delta Rising
#define MPR121_NCLR      0x2D  // Noise Count Limit Rising
#define MPR121_FDLR      0x2E  // Filter Delay Count Rising
#define MPR121_MHDF      0x2F  // Max Half Delta Falling
#define MPR121_NHDF      0x30  // Noise Half Delta Falling
#define MPR121_NCLF      0x31  // Noise Count Limit Falling
#define MPR121_FDLF      0x32  // Filter Delay Count Falling
#define MPR121_NHDT      0x33  // Noise Half Delta Touched
#define MPR121_NCLT      0x34  // Noise Count Limit Touched
#define MPR121_FDLT      0x35  // Filter Delay Count Touched
#define MPR121_DEBOUNCE  0x5B  // Debounce register
#define MPR121_AUTOCONF0 0x7B  // Auto-configuration control 0
#define MPR121_AUTOCONF1 0x7C  // Auto-configuration control 1
#define MPR121_USL       0x7D  // Upper search limit
#define MPR121_LSL       0x7E  // Lower search limit
#define MPR121_TL        0x7F  // Target level

// Write a single byte to an MPR121 register
void mpr121Write(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(MPR121_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

// Configure MPR121 for reliable touch detection
// Must be called after cap.begin() succeeds
void configureMPR121() {
    // Stop electrodes before changing config
    mpr121Write(MPR121_ECR, 0x00);

    // Baseline tracking parameters (more responsive)
    mpr121Write(MPR121_MHDR, 0x01);  // Max half delta rising
    mpr121Write(MPR121_NHDR, 0x01);  // Noise half delta rising
    mpr121Write(MPR121_NCLR, 0x0E);  // Noise count limit rising
    mpr121Write(MPR121_FDLR, 0x00);  // Filter delay rising
    mpr121Write(MPR121_MHDF, 0x01);  // Max half delta falling
    mpr121Write(MPR121_NHDF, 0x05);  // Noise half delta falling
    mpr121Write(MPR121_NCLF, 0x01);  // Noise count limit falling
    mpr121Write(MPR121_FDLF, 0x00);  // Filter delay falling
    mpr121Write(MPR121_NHDT, 0x00);  // Noise half delta touched
    mpr121Write(MPR121_NCLT, 0x00);  // Noise count limit touched
    mpr121Write(MPR121_FDLT, 0x00);  // Filter delay touched

    // Debounce: 2 samples for both touch and release
    mpr121Write(MPR121_DEBOUNCE, 0x22);

    // Increase charge current (CONFIG1)
    // Bits 5:0 = CDC (charge/discharge current), max 63µA
    // Default library sets 0x10 = 16µA, boost to 63µA
    mpr121Write(MPR121_CONFIG1, 0x3F);

    // Charge time + sampling (CONFIG2)
    // Bits 7:5 = CDT (charge time): 0b100 = 1µs
    // Bits 4:3 = SFI (samples): 0b00 = 4 samples
    // Bits 2:0 = ESI (period): 0b100 = 16ms
    mpr121Write(MPR121_CONFIG2, 0x84);

    // Set touch/release thresholds
    cap.setThresholds(TOUCH_THRESHOLD, RELEASE_THRESHOLD);

    // Enable auto-configuration for automatic baseline calibration
    // USL = (Vdd - 0.7V) / Vdd * 256 = (3.3 - 0.7) / 3.3 * 256 ≈ 201
    mpr121Write(MPR121_USL, 201);
    mpr121Write(MPR121_LSL, 131);  // 65% of USL
    mpr121Write(MPR121_TL,  181);  // 90% of USL
    // Auto-config: enable, auto-reconfigure on out-of-range
    mpr121Write(MPR121_AUTOCONF0, 0x0B);
    mpr121Write(MPR121_AUTOCONF1, 0x00);

    // Re-enable electrodes (12 electrodes, CL=10 for baseline tracking)
    mpr121Write(MPR121_ECR, 0x8C);

    // Let auto-calibration settle
    delay(200);
}

// ---------- Display Task (Core 0) ----------

TaskHandle_t drawTaskHandle;

void drawTask(void *parameter) {
    while (true) {
        bool curSensorOK;
        bool curTouched;
        uint16_t curChannels;
        uint16_t curFiltered;
        uint16_t curBaseline;

        if (xSemaphoreTake(xMutex, portMAX_DELAY)) {
            curSensorOK = disp_sensorOK;
            curTouched  = disp_touched;
            curChannels = disp_channels;
            curFiltered = disp_filtered;
            curBaseline = disp_baseline;
            xSemaphoreGive(xMutex);
        } else {
            vTaskDelay(10);
            continue;
        }

        // Clear
        canvas->fillScreen(COLOR_BG);

        // --- Title ---
        canvas->setTextColor(COLOR_LABEL);
        canvas->setTextSize(2, 2, 1);
        canvas->setCursor(20, 15);
        canvas->print("Touch Sensor");

        // --- Sensor status ---
        canvas->setTextSize(1, 1, 1);
        canvas->setCursor(15, 45);
        canvas->setTextColor(COLOR_LABEL);
        canvas->print("MPR121: ");
        if (curSensorOK) {
            canvas->setTextColor(COLOR_TOUCHED);
            canvas->print("Connected");
        } else {
            canvas->setTextColor(COLOR_ERROR);
            canvas->print("Unplugged");
        }

        if (curSensorOK) {
            // --- Big touch indicator circle ---
            uint16_t circleColor = curTouched ? COLOR_TOUCHED : COLOR_IDLE;
            canvas->fillCircle(CIRCLE_X, CIRCLE_Y, CIRCLE_R, circleColor);

            // Label inside circle
            canvas->setTextSize(3, 3, 1);
            canvas->setTextColor(COLOR_BG);
            int labelX = curTouched ? 37 : 25;
            canvas->setCursor(labelX, CIRCLE_Y - 8);
            canvas->print(curTouched ? "TOUCH" : "READY");

            // --- Raw sensor data (debug) ---
            canvas->setTextSize(1, 1, 1);
            canvas->setTextColor(COLOR_TEXT);
            canvas->setCursor(15, 235);
            canvas->printf("Filt: %4d  Base: %4d", curFiltered, curBaseline);
            int16_t delta = (int16_t)curBaseline - (int16_t)curFiltered;
            canvas->setCursor(15, 250);
            canvas->printf("Delta: %4d  Thr: %d", delta, TOUCH_THRESHOLD);

            // --- Channel + status ---
            canvas->setTextSize(2, 2, 1);
            canvas->setCursor(15, 275);
            canvas->printf("Chan: 0x%03X", curChannels & 0x0FFF);

            canvas->setCursor(15, 300);
            if (curTouched) {
                canvas->setTextColor(COLOR_TOUCHED);
                canvas->print("Touching");
            } else {
                canvas->setTextColor(COLOR_LABEL);
                canvas->print("Released");
            }
        } else {
            // --- Error state ---
            canvas->fillCircle(CIRCLE_X, CIRCLE_Y, CIRCLE_R, COLOR_ERROR);
            canvas->setTextSize(2, 2, 1);
            canvas->setTextColor(COLOR_BG);
            canvas->setCursor(34, CIRCLE_Y - 8);
            canvas->print("ERROR");

            canvas->setTextSize(1, 1, 1);
            canvas->setTextColor(COLOR_LABEL);
            canvas->setCursor(15, 245);
            canvas->print("Check I2C wiring");
            canvas->setCursor(15, 260);
            canvas->print("Retrying...");
        }

        canvas->flush();
        vTaskDelay(100);  // ~10Hz refresh
    }
}

// ---------- Setup ----------

void setup() {
    GFX_EXTRA_PRE_INIT();

    #ifdef GFX_BL
    pinMode(GFX_BL, OUTPUT);
    digitalWrite(GFX_BL, HIGH);
    #endif

    Serial.begin(115200);

    // Initialize display
    if (!gfx->begin()) {
        Serial.println("gfx->begin() failed!");
    }
    canvas = new Arduino_Canvas(SCREEN_W, SCREEN_H, gfx, 0, 0);
    canvas->begin(GFX_SKIP_OUTPUT_BEGIN);
    canvas->fillScreen(COLOR_BG);
    canvas->setTextColor(COLOR_LABEL);
    canvas->setTextSize(2, 2, 1);
    canvas->setCursor(20, 150);
    canvas->print("Starting...");
    canvas->flush();

    // Initialize I2C on STEMMA QT pins
    Wire.begin(I2C_SDA, I2C_SCL);

    // Initialize MPR121
    sensorOK = cap.begin(MPR121_ADDR, &Wire);
    if (sensorOK) {
        configureMPR121();
    }

    // Create mutex and display task
    xMutex = xSemaphoreCreateMutex();

    // Push initial state to display
    if (xSemaphoreTake(xMutex, portMAX_DELAY)) {
        disp_sensorOK = sensorOK;
        disp_touched = false;
        disp_channels = 0;
        xSemaphoreGive(xMutex);
    }

    xTaskCreatePinnedToCore(
        drawTask,        // Task function
        "Draw Task",     // Name
        10000,           // Stack size (bytes)
        NULL,            // Parameter
        1,               // Priority
        &drawTaskHandle, // Handle
        0                // Core 0
    );
}

// ---------- Helpers ----------

// Returns true if the MPR121 responds on I2C
bool pingSensor() {
    Wire.beginTransmission(MPR121_ADDR);
    return Wire.endTransmission() == 0;
}

void markDisconnected(unsigned long now) {
    sensorOK = false;
    lastReconnectTime = now;
    if (xSemaphoreTake(xMutex, portMAX_DELAY)) {
        disp_sensorOK = false;
        disp_touched = false;
        disp_channels = 0;
        xSemaphoreGive(xMutex);
    }
}

// ---------- Main Loop (Core 1) ----------

void loop() {
    unsigned long now = millis();

    // Check for serial commands
    while (Serial.available()) {
        char cmd = Serial.read();
        if (cmd == 'C') {
            calibrationMode = true;
        } else if (cmd == 'N') {
            calibrationMode = false;
        }
    }

    // Rate limit serial output
    if (now - lastSendTime < SEND_INTERVAL_MS) {
        return;
    }
    lastSendTime = now;

    // Handle disconnected sensor
    if (!sensorOK) {
        Serial.println("E01");

        // Periodically attempt reconnection
        if (now - lastReconnectTime >= RECONNECT_INTERVAL_MS) {
            lastReconnectTime = now;
            if (pingSensor()) {
                sensorOK = cap.begin(MPR121_ADDR, &Wire);
                if (sensorOK) {
                    configureMPR121();
                }
            }
            if (xSemaphoreTake(xMutex, portMAX_DELAY)) {
                disp_sensorOK = sensorOK;
                disp_touched = false;
                disp_channels = 0;
                xSemaphoreGive(xMutex);
            }
        }
        return;
    }

    // Periodically verify sensor is still on the bus
    if (now - lastPingTime >= PING_INTERVAL_MS) {
        lastPingTime = now;
        if (!pingSensor()) {
            Serial.println("E02");
            markDisconnected(now);
            return;
        }
    }

    // Read touch status from MPR121
    uint16_t touched = cap.touched();

    // 0xFFFF means I2C read failed
    if (touched == 0xFFFF) {
        Serial.println("E02");
        markDisconnected(now);
        return;
    }

    // Read raw electrode data for active channel (for debug display + calibration)
    uint16_t filtered = cap.filteredData(ACTIVE_CHANNEL);
    uint16_t baseline = cap.baselineData(ACTIVE_CHANNEL);

    if (calibrationMode) {
        // Calibration output: C<filtered>,<baseline>,<touched_hex>
        char cbuf[32];
        snprintf(cbuf, sizeof(cbuf), "C%d,%d,%03X", filtered, baseline, touched & 0x0FFF);
        Serial.println(cbuf);
    } else {
        // Normal output: T<hex>
        char buf[8];
        snprintf(buf, sizeof(buf), "T%03X", touched & 0x0FFF);
        Serial.println(buf);
    }

    // Update display state
    bool isTouched = (touched & 0x0FFF) != 0;
    if (xSemaphoreTake(xMutex, portMAX_DELAY)) {
        disp_sensorOK = true;
        disp_touched = isTouched;
        disp_channels = touched;
        disp_filtered = filtered;
        disp_baseline = baseline;
        xSemaphoreGive(xMutex);
    }

    lastTouched = touched;
}
