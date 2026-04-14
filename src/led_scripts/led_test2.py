import time
import gpiod
from gpiod.line import Direction, Value

CHIP = "/dev/gpiochip0"
OFFSET = 144  # Pin 7 = gpio492

with gpiod.request_lines(
    CHIP,
    consumer="led",
    config={
        OFFSET: gpiod.LineSettings(
            direction=Direction.OUTPUT,
            output_value=Value.ACTIVE,  # HIGH = LED OFF
        )
    },
) as req:
    print("LED ON")
    req.set_value(OFFSET, Value.INACTIVE)  # LOW -> LED ON
    time.sleep(2)

    print("LED OFF")
    req.set_value(OFFSET, Value.ACTIVE)    # HIGH -> LED OFF
    time.sleep(2)