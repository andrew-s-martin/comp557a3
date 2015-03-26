# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:11:25 2015

@author: fmannan
"""
import os
import fnmatch
from SceneParser import SceneParser

# maximum dimension used for rendering all the scenes
# set the maximum dimension to something suitable for faster rendering
MAXDIM = 128  #2048 #256  # 64

# output directory relative path
OUTDIR_REL_PATH = './images_test_render/'

# dataset directory
DATASET_DIR = './scenes/'

filterStr = '*.xml' # xml file filter pattern

if not os.path.exists(OUTDIR_REL_PATH): 
        os.makedirs(OUTDIR_REL_PATH)
        
for root, subDirs, files in os.walk(DATASET_DIR):
    
    scene_files = fnmatch.filter(os.listdir(root), filterStr)
    
    for scene_file in scene_files:
        scene_file_path = os.path.join(root, scene_file)
        print(scene_file_path)
        try:
            scene = SceneParser(scene_file_path).scene
            scene.render.OUTDIR_REL_PATH = OUTDIR_REL_PATH
            scene.render.bShowImage = False
            camera = scene.render.camera 
            print(camera.imageSize, max(camera.imageSize))
            scale_factor = min(float(MAXDIM) / max(camera.imageSize), 1.)
            camera.imageWidth = int(camera.imageWidth * scale_factor)
            camera.imageHeight = int(camera.imageHeight * scale_factor)
            camera.imageSize = (camera.imageWidth, camera.imageHeight)
            print(scene.render.camera.imageSize)
            scene.renderScene()
        except:
            print('failed to render ' + scene_file + ' moving on...')
