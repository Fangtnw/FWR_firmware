#include "fwr_control.h"
#include "sbus_rx.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern "C" void app_main(void)
{
    /* ---- Initialize flight control (PWM + timers) ---- */
    fwr_control_init();
    fwr_control_start();

    /* ---- Initialize SBUS receiver (Futaba T12K) ---- */
    sbus_rx_init();
    sbus_rx_start();

    /* ---- Idle loop ---- */
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
