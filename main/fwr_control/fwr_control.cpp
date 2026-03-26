#include "fwr_control.h"

#include "driver/ledc.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <math.h>
#include <string.h>

/* ===================== Constants ===================== */


#define PI 3.141592653589793f

// ---- Servo config ----
#define SERVO_LEFT_PIN    4
#define SERVO_RIGHT_PIN   5

#define SERVO_LEFT_CH     LEDC_CHANNEL_0
#define SERVO_RIGHT_CH    LEDC_CHANNEL_1

#define SERVO_TIMER       LEDC_TIMER_0
#define SERVO_MODE        LEDC_LOW_SPEED_MODE
#define SERVO_FREQ_HZ     250

#define SERVO_US_MIN      900
#define SERVO_US_MAX      2100
#define SERVO_US_NEUTRAL  1500

// ---- Timing ----
static const int dt_sin_ms   = 3;
static const int dt_servo_ms = 4;

/* ===================== Control Variables ===================== */

// (to be updated later by SBUS task)
volatile int aileron = 0; //roll
volatile int elevator = 0; //pitch
volatile int throttle = 0;
volatile int rudder = 0; //yaw
volatile int frequency = 0;
volatile int servo_adj_l = 0;
volatile int servo_adj_r = 0;
volatile int flag_square_wave = 0;

// OFD obstacle avoidance offsets (added to SBUS aileron/rudder)
volatile int ofd_aileron_offset = 0;
volatile int ofd_rudder_offset  = 0;

// Sin generator state
static float y = 0.0f;
static float y_old = 0.0f;
static float y_cos = 0.0f;
static float y_cos_old = 0.0f;
static float phi = 0.0f;
static int   freq_tmp = 0;
static int64_t t_record = 0;

// Protect shared float
static portMUX_TYPE y_mux = portMUX_INITIALIZER_UNLOCKED;

/* ===================== Utility ===================== */

static inline int clamp_us(int us)
{
    if (us < SERVO_US_MIN) return SERVO_US_MIN;
    if (us > SERVO_US_MAX) return SERVO_US_MAX;
    return us;
}

static uint32_t us_to_duty(uint32_t us)
{
    uint32_t period_us = 1000000UL / SERVO_FREQ_HZ;
    return (us * ((1UL << 16) - 1)) / period_us;
}

static inline void servo_write(ledc_channel_t ch, int us)
{
    ledc_set_duty(SERVO_MODE, ch, us_to_duty(us));
    ledc_update_duty(SERVO_MODE, ch);
}

/* ===================== Timers ===================== */

static void IRAM_ATTR sin_timer_cb(void *arg)
{
    int64_t now_ms = esp_timer_get_time() / 1000;

    if (freq_tmp != frequency) {
        phi = atan2f(y_old, y_cos_old);
        freq_tmp = frequency;
        t_record = now_ms - dt_sin_ms;
    }

    float t = (now_ms - t_record) * 0.001f;
    float w = 2.0f * PI * (freq_tmp / 100.0f);

    float y_new     = sinf(phi + w * t);
    float y_cos_new = cosf(phi + w * t);

    if (flag_square_wave) {
        y_new = (y_new >= 0.0f) ? 1.0f : -1.0f;
    }

    portENTER_CRITICAL_ISR(&y_mux);
    y = y_new;
    y_old = y_new;
    y_cos = y_cos_new;
    y_cos_old = y_cos_new;
    portEXIT_CRITICAL_ISR(&y_mux);
}

static void IRAM_ATTR servo_timer_cb(void *arg)
{
    float y_local;

    portENTER_CRITICAL_ISR(&y_mux);
    y_local = y;
    portEXIT_CRITICAL_ISR(&y_mux);

    int ail = aileron + ofd_aileron_offset;
    int rud = rudder + ofd_rudder_offset;

    int left =
        SERVO_US_NEUTRAL +
        (-elevator + ail + servo_adj_l) +
        (throttle + rud) * y_local;

    int right =
        SERVO_US_NEUTRAL +
        ( elevator + ail + servo_adj_r) +
        (-throttle + rud) * y_local;

    servo_write(SERVO_LEFT_CH,  clamp_us(left));
    servo_write(SERVO_RIGHT_CH, clamp_us(right));
}

/* ===================== Public API ===================== */

void fwr_control_init(void)
{
    /* ---- LEDC timer ---- */
    ledc_timer_config_t timer = {};
    timer.speed_mode       = SERVO_MODE;
    timer.timer_num        = SERVO_TIMER;
    timer.freq_hz          = SERVO_FREQ_HZ;
    timer.duty_resolution  = LEDC_TIMER_16_BIT;
    timer.clk_cfg          = LEDC_AUTO_CLK;
    ledc_timer_config(&timer);

    /* ---- Left servo ---- */
    ledc_channel_config_t ch0 = {};
    ch0.channel    = SERVO_LEFT_CH;
    ch0.gpio_num   = SERVO_LEFT_PIN;
    ch0.speed_mode = SERVO_MODE;
    ch0.timer_sel  = SERVO_TIMER;
    ch0.duty       = 0;
    ch0.hpoint     = 0;
    ledc_channel_config(&ch0);

    /* ---- Right servo ---- */
    ledc_channel_config_t ch1 = ch0;
    ch1.channel  = SERVO_RIGHT_CH;
    ch1.gpio_num = SERVO_RIGHT_PIN;
    ledc_channel_config(&ch1);

    /* ---- Safe neutral output ---- */
    servo_write(SERVO_LEFT_CH,  SERVO_US_NEUTRAL);
    servo_write(SERVO_RIGHT_CH, SERVO_US_NEUTRAL);
}

void fwr_control_start(void)
{
    static esp_timer_handle_t sin_timer;
    static esp_timer_handle_t servo_timer;

    esp_timer_create_args_t sin_args = {};
    sin_args.callback = sin_timer_cb;
    sin_args.name = "sin_timer";
    sin_args.dispatch_method = ESP_TIMER_TASK;
    sin_args.skip_unhandled_events = true;
    esp_timer_create(&sin_args, &sin_timer);
    esp_timer_start_periodic(sin_timer, dt_sin_ms * 1000);

    esp_timer_create_args_t servo_args = {};
    servo_args.callback = servo_timer_cb;
    servo_args.name = "servo_timer";
    servo_args.dispatch_method = ESP_TIMER_TASK;
    servo_args.skip_unhandled_events = true;
    esp_timer_create(&servo_args, &servo_timer);
    esp_timer_start_periodic(servo_timer, dt_servo_ms * 1000);
}

void fwr_set_ofd_avoidance(int aileron_offset_us, int rudder_offset_us)
{
    ofd_aileron_offset = aileron_offset_us;
    ofd_rudder_offset  = rudder_offset_us;
}
