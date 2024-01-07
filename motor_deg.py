from pybricks.pupdevices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait

# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll

m1 = Motor(Port.A)
m2 = Motor(Port.B)

# Optional: Register stdin for polling. This allows
# you to wait for incoming data without blocking.
keyboard = poll()
keyboard.register(stdin)

m1.reset_angle(180)
m2.reset_angle(0)

while True:
    # Let the remote program know we are ready for a command.
    stdout.buffer.write(b"rdy")

    # Optional: Check available input.
    while not keyboard.poll(0):
        # Optional: Do something here.
        wait(10)

    # Read three bytes.
    cmd = stdin.buffer.read(4)

    # Decide what to do based on the command.
    motor = int(cmd[0])
    if cmd == b"":
        motor.dc(50)
    elif cmd == b"rev":
        motor.dc(-50)
    elif cmd == b"bye":
        break
    else:
        motor.stop()

