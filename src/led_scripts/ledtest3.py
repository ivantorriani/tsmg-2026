import gpiod
import time

CHIP_NAME = "gpiochip0"
GPIO_LINE = 105

chip = gpiod.Chip(CHIP_NAME)
lines = chip.get_lines([GPIO_LINE])

lines.request(consumer="toggle_led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])

while True:
    lines.set_values([1])
    time.sleep(1)
    lines.set_values([0])
    time.sleep(1)
    print("cycle")