# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 16:32:41 2015

@author: fmannan
"""
from __future__ import division
import numpy as np
from numpy import deg2rad, cos, sin, sqrt

def normalize(vec):
    vec = np.array(vec)
    n = np.linalg.norm(vec)
    if abs(n) > 1e-12:
        vec = vec / n
    return vec

def scale(scale_vec):
    S = np.eye(4)
    S[:3,:3] = np.diag(np.array(scale_vec))
    return np.matrix(S)
    
def translate(translate_vec):
    T = np.eye(4);
    T[:3,3] = translate_vec
    return np.matrix(T)

def rotX(c, s):
    R = [[1, 0, 0, 0], 
         [0, c, -s, 0], 
         [0, s, c, 0],
         [0, 0, 0, 1]]
    return np.matrix(R)

def rotY(c, s):
    R = [[c, 0, s, 0], 
         [0, 1, 0, 0],
         [-s, 0, c,  0], 
         [0, 0, 0, 1] ]
    return np.matrix(R)

def rotZ(c, s):
    R = [[c, -s, 0, 0], 
         [s,  c, 0, 0], 
         [0, 0, 1, 0],
         [0, 0, 0, 1] ]
    return np.matrix(R)
    
def rotateX(angle_deg):
    theta = deg2rad(angle_deg)
    return rotX(cos(theta), sin(theta))

def rotateY(angle_deg):
    theta = deg2rad(angle_deg)
    return rotY(cos(theta), sin(theta))

def rotateZ(angle_deg):
    theta = deg2rad(angle_deg)
    return rotZ(cos(theta), sin(theta))
    
def rotate(angle_deg, axis):
    axis = normalize(axis)
    u = sqrt(axis[0]*axis[0] + axis[2]*axis[2])
    if (abs(u) < 1e-8): 
	return rotateY(np.sign(axis[1]) * angle_deg)
	
    cosphi = u
    sinphi = axis[1]
    costh = axis[2] / u
    sinth = axis[0] / u
    #for simplicity just multiply the matrices explicitly
    return rotY(costh, sinth) * rotX(cosphi, -sinphi) * rotateZ(angle_deg) * rotX(cosphi, sinphi) * rotY(costh, -sinth);
    
def lookAtMatrix(eye, center, up):
    eye = np.array(eye)
    lookat = normalize(np.array(center) - eye)
    up = normalize(np.array(up))
    
    cam_right = normalize(np.cross(lookat, up))
    cam_up = normalize(np.cross(cam_right, lookat))
    M = np.eye(4)
    M[0,:3] = cam_right
    M[1,:3] = cam_up
    M[2,:3] = -lookat
    M[:3,3] = np.dot(M[:3,:3], -eye)
    return np.matrix(M)

def gluLookAtMatrix(eye, center, up):
    '''
    get the lookat matrix using opengl. Note that opengl context needs to be
    initialized at the point of making this call.
    '''
    from OpenGL.GL import GL_MODELVIEW, GL_MODELVIEW_MATRIX, glMatrixMode, \
                            glLoadIdentity, glGetDoublev, glPushMatrix, \
                            glPopMatrix
    from OpenGL.GLU import gluLookAt
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    gluLookAt(eye[0], eye[1], eye[2], 
              center[0], center[1], center[2], 
              up[0], up[1], up[2]) 
    M = np.transpose(glGetDoublev(GL_MODELVIEW_MATRIX))
    glPopMatrix()
    return np.matrix(M)
    
def perspectiveMatrix(fovy, aspect, zNear, zFar):
    f = 1.0/np.tan(np.deg2rad(fovy)/2)
    # perspective transformation matrix
    P = np.zeros((4, 4))
    P[0][0] = f/aspect
    P[1][1] = f
    P[2][2] = (zNear + zFar) / (zNear - zFar) #-1.0/zNear;
    P[2][3] = 2.0 * zNear * zFar / (zNear - zFar) #zNear / (zNear - zFar);
    P[3][2] = -1.0
    
    return np.matrix(P)

