/* pin.c -- the test pins' storage and setter (exl3i_pin.h). */
#include "exl3i_pin.h"

int64_t exl3i_pins[EXL3I_PIN_N];

void exl3i_test_pin(int pin, int64_t value) {
    if (pin >= 0 && pin < EXL3I_PIN_N) exl3i_pins[pin] = value;
}
