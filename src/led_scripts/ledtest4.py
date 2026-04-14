import Jetson.GPIO as GPIO
import time

led_pin = 7

GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)

try:
    while True:
        GPIO.output(led_pin, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(led_pin, GPIO.LOW)
        time.sleep(1)
except KeyboardInterrupt:
    print("fuck this shit")
    GPIO.cleanup()