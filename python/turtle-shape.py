#!/usr/bin/env python3

import turtle
import time
import sys

sides = input("how many sides do you want on your regular polyhedron\n")

sides = int(sides)

angle = 360/sides

print("what do you want the perimeter of the shape to be? (in pixels)")
perimeter = int(sys.stdin.readline())
edge_length = round(perimeter/sides)
print(edge_length)
t = turtle.pen()
for x in range(0,sides):
    turtle.forward(edge_length)
    turtle.right(angle)

time.sleep(5)
