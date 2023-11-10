import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import os
import uuid

verticies = ((1,-1,-1,),(1,1,-1),(-1,1,-1),(-1,-1,-1),(1,-1,1),(1,1,1),(-1,-1,1),(-1,1,1))

edges = ((0,1),(0,3),(0,4),(2,1),(2,3),(2,7),(6,3),(6,4),(6,7),(5,1),(5,4),(5,7))

surfaces = ((0,1,2,3),(3,2,7,6),(6,7,5,4),(4,5,1,0),(1,5,7,2),(4,0,3,6))

colours = ((1,1,1,0),(1,0,0,1),(0,0,1,1),(1,1,0,1)) 

#("nothing":1,1,1,,0 "red":1,0,0,1 "blue":0,0,1,1 "yellow":1,1,0,1)

def Colour():
    for colour in colours:
        glColor4f(colour)
        
def Cube1():
    glBegin(GL_QUADS)
    for surface in surfaces:
        glColor4f(1,0,0,1) #hier kommt Farbcode rein
        for vertex in surface:
            glVertex3fv(verticies[vertex])
    glEnd()
    
    glBegin(GL_LINES)
    for edge in edges:
        glColor4f(0,0,0,1)
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()
    
def Cube2():
    glBegin(GL_QUADS)
    for surface in surfaces:
        glColor4f(1,0,0,1) #hier kommt Farbcode rein
        for vertex in surface:
            glVertex3fv(verticies[vertex])
    glEnd()
    
    glBegin(GL_LINES)
    for edge in edges:
        glColor4f(0,0,0,1)
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()
    
def Cube3():
    glBegin(GL_QUADS)
    for surface in surfaces:
        glColor4f(1,0,0,1) #hier kommt Farbcode rein
        for vertex in surface:
            glVertex3fv(verticies[vertex])
    glEnd()
    
    glBegin(GL_LINES)
    for edge in edges:
        glColor4f(0,0,0,1)
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()
    
def Cube4():
    glBegin(GL_QUADS)
    for surface in surfaces:
        glColor4f(1,0,0,1) #hier kommt Farbcode rein
        for vertex in surface:
            glVertex3fv(verticies[vertex])
    glEnd()
    
    glBegin(GL_LINES)
    for edge in edges:
        glColor4f(0,0,0,1)
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()
    
        
    
#funktioniert noch nicht
    
def save_image(filename):
    glReadBUffer(GL_FRONT)
    data = glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (width,heigth), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(filename, "PNG")

def main():
    pygame.init()
    display = (800,800)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    gluPerspective(100, display[0]/display[1], 0.1, 100)
    gluLookAt(0,3,3,0,0,-1,0,0,-1)
    glTranslatef(0.0,0.0,-5)
    glRotatef(20, 0, -5, 0)
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glClearColor(1,1,1,1)
        glPushMatrix()
        glTranslatef(-1,-1,1)
        Cube1()
        glPopMatrix()
        glPushMatrix()
        glTranslatef(1,-1,1)
        Cube2()
        glPopMatrix()
        pygame.display.flip()
        pygame.time.wait(10)
main()

# Generiert universally unique identifier

import os
import uuid


labels = [options] #definieren oder löschen

IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images')

if not os.path.exists(IMAGES_PATH):
    if os.name == "nt":
        !mkdir {IMAGES_PATH}
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        !mkdir {path}
    imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1()))) #muss noch definiert werden
    save_image(imagename)

# Möglichkeiten Generator

def generate_options(Colours,Cubes, prefix=None, results=None):

    if prefix is None:
        prefix = []
    if results is None:
        results = []
        
    if Cubes == 0:
        results.append(prefix)
        return
        
    for Colour in Colours:
        new_prefix = prefix + [Colour]
        generate_options(Colours, Cubes - 1, new_prefix, results)
        
    return results

Colours = [0,1,2,3]
Cubes = 2
result = generate_options(Colours,Cubes)

x = []
for options in result:
    print(options)
    x = [x+options] #muss noch umgeschrieben werden damit eine Liste