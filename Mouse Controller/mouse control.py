# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:12:49 2020

@author: lenovo
"""

#pip install pynput

from pynput.mouse import Button, Controller


mouse = Controller()

print ("Current position: " + str(mouse.position))


mouse.position = (10, 20)
mouse.click(Button.left, 1)
mouse.click(Button.right, 1)
