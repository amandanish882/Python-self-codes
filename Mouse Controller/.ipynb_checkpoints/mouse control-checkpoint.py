# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:12:49 2020

@author: lenovo
"""

#pip install pynput

from pynput.mouse import Button, Controller

# Getting screen resolution
#from win32api import GetSystemMetrics

#print("Width =", GetSystemMetrics(0))
#print("Height =", GetSystemMetrics(1))

mouse = Controller()

print ("Current position: " + str(mouse.position))

time.sleep(2.4)

mouse.position = (710, 694)
mouse.click(Button.left, 1)
mouse.click(Button.right, 1)

# Window position(710, 694)
# Resolution : Width = 1280, Height = 720

import time

#unit is in seconds

mouse.position = (10, 20)



