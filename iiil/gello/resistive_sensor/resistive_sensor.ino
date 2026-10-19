#include <Arduino.h>
#include "Arduino_GFX_Library.h"

#define GFX_DEV_DEVICE LILYGO_T_DISPLAY_S3
#define GFX_EXTRA_PRE_INIT()              \
    {                                     \
        pinMode(15 /* PWD */, OUTPUT);    \
        digitalWrite(15 /* PWD */, HIGH); \
    }
#define GFX_BL 38

Arduino_DataBus *bus = new Arduino_ESP32PAR8Q(
    7 /* DC */, 6 /* CS */, 8 /* WR */, 9 /* RD */,
    39 /* D0 */, 40 /* D1 */, 41 /* D2 */, 42 /* D3 */, 45 /* D4 */, 46 /* D5 */, 47 /* D6 */, 48 /* D7 */);
Arduino_GFX *gfx = new Arduino_ST7789(bus, 5 /* RST */, 0 /* rotation */, true /* IPS */, 170 /* width */, 320 /* height */, 35 /* col offset 1 */, 0 /* row offset 1 */, 35 /* col offset 2 */, 0 /* row offset 2 */);

Arduino_GFX *gfx2;

TaskHandle_t drawTaskHandle;

float Vsense_1 = 0.0;
float Vsense_2 = 0.0;
float emaVsense_1 = 0.0;  // EMA for Vsense_1
float emaVsense_2 = 0.0;  // EMA for Vsense_2
float alpha = 0.1;        // Smoothing factor for EMA


SemaphoreHandle_t xMutex;

unsigned long lastPrintTime = 0;  // Stores the time of the last print
const unsigned long printInterval = 2;  // Interval in milliseconds for serial printing

void setBrightness(uint8_t value)
{
    bus->beginWrite();
    bus->writeC8D8(0x51, value);
    bus->endWrite();
}

void drawTask(void *parameter)
{
    int radius_factor = 1200;
    int max_radius = 100;
    float text_sz = 2;

    while (true)  // Keep the task running indefinitely
    {
        if (xSemaphoreTake(xMutex, portMAX_DELAY))  // Take the mutex before accessing shared variables
        {
            float Vs1_temp = emaVsense_1;
            float Vs2_temp = emaVsense_2;
            xSemaphoreGive(xMutex);  // Release the mutex

            gfx2->fillRect(0, 0, 170, 320, BLACK);

            int radius_1 = map(Vs1_temp * radius_factor, 0, 3300, 1, max_radius);
            gfx2->fillCircle(85, 75, radius_1, GREEN);

            int radius_2 = map(Vs2_temp * radius_factor, 0, 3300, 1, max_radius);
            gfx2->fillCircle(85, 245, radius_2, GREEN);

            gfx2->setCursor(20, 145);
            gfx2->setTextSize(text_sz, text_sz, 1);
            gfx2->printf("V1: %.3f V   V2: %.3f V", Vs1_temp, Vs2_temp);

            gfx2->flush();  // Push the canvas content to the screen
        }

        vTaskDelay(1);  // Yield to other tasks, adjust this delay if necessary
    }
}

void setup()
{
    GFX_EXTRA_PRE_INIT();

    #ifdef GFX_BL
    pinMode(GFX_BL, OUTPUT);
    digitalWrite(GFX_BL, HIGH);
    #endif

    Serial.begin(115200);

    if (!gfx->begin()) {
        Serial.println("gfx->begin() failed!");
    }

    gfx2 = new Arduino_Canvas(170, 320, gfx, 0, 0);
    gfx2->begin(GFX_SKIP_OUTPUT_BEGIN);
    gfx2->fillScreen(BLACK);
    gfx2->setCursor(40, 160);
    gfx2->setTextColor(RED);
    gfx2->setTextSize(2, 2, 1);
    gfx2->println("Loading");
    gfx2->fillCircle(85, 85, 25, GREEN);
    gfx2->flush();  // Initial flush to show "Loading"

    // Create a mutex before starting the tasks
    xMutex = xSemaphoreCreateMutex();

    xTaskCreatePinnedToCore(
        drawTask,    // Task function
        "Draw Task", // Task name
        10000,       // Stack size (bytes)
        NULL,        // Task input parameter
        1,           // Task priority
        &drawTaskHandle,  // Task handle
        0            // Core to run the task (Core 0)
    );
}

void loop()
{
    if (xSemaphoreTake(xMutex, portMAX_DELAY))  // Take the mutex before accessing shared variables
    {
        // Read sensor values as fast as possible
        int sensorValue_1 = analogRead(1);
        Vsense_1 = sensorValue_1 * (3.3 / 4095.0);
        Vsense_1 = max(0.0f, min(Vsense_1, 2.0f));

        int sensorValue_2 = analogRead(2);
        Vsense_2 = sensorValue_2 * (3.3 / 4095.0);
        Vsense_2 = max(0.0f, min(Vsense_2, 2.0f));

        // Apply Exponential Moving Average (EMA) smoothing
        emaVsense_1 = (alpha * Vsense_1) + ((1 - alpha) * emaVsense_1);
        emaVsense_2 = (alpha * Vsense_2) + ((1 - alpha) * emaVsense_2);

        xSemaphoreGive(xMutex);  // Release the mutex
    }

    // Check if 2 milliseconds have passed since the last print
    unsigned long currentTime = millis();
    if (currentTime - lastPrintTime >= printInterval)
    {
        // Print to serial console
        Serial.println(emaVsense_1, 4);
        // Serial.print(",");
        // Serial.println(emaVsense_2, 4);

        // Update the last print time
        lastPrintTime = currentTime;
    }

    // No delay here, so the loop runs as fast as possible
}