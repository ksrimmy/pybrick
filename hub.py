from pybricks.pupdevices import Motor
from pybricks.parameters import Port, Stop
from pybricks.tools import wait

# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll

m1 = Motor(Port.C)
m2 = Motor(Port.D)
m3 = Motor(Port.B)

# Optional: Register stdin for polling. This allows
# you to wait for incoming data without blocking.
keyboard = poll()
keyboard.register(stdin)

# m1.reset_angle(180)
# m2.reset_angle(0)
cmd = ""

while True:
    # Let the remote program know we are ready for a command.
    stdout.buffer.write(b"rdy")

    # Optional: Check available input.
    while not keyboard.poll(0):
        # Optional: Do something here.
        wait(10)

    if keyboard.poll(0):
        #cmd = input()
        cmd += stdin.read(1)
    
        if cmd[-1] == "\n":
            cmd_arr = cmd[0:-1].split("|")

            m1.track_target(int(cmd_arr[0]))
            m2.track_target(int(cmd_arr[1]))
            m3.track_target(int(cmd_arr[2]))
            
            #m2.run_target(500, deg, then=Stop.COAST)
            
            cmd = ""
