import Jetson.GPIO as GPIO
import time
import curses

led_pin1 = 7
led_pin2 = 15

GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(led_pin2, GPIO.OUT, initial=GPIO.LOW)

def main(stdscr):
    try:
        GPIO.output(led_pin, GPIO.LOW)
        stdscr.keypad(True) #access keyboard
        while True:
            key = stdscr.getch()
            if (key == curses.KEY_UP):
                GPIO.output(led_pin, GPIO.HIGH)
            elif (key == curses.KEY_DOWN):
                GPIO.output(led_pin, GPIO.LOW)
    except KeyboardInterrupt:
        print("exiting")
        GPIO.cleanup()

if __name__ == "__main__":
    curses.wrapper(main)