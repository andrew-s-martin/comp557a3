'''
A3App.py

Usage: python A3App.py path_to_scene_file
example: python A3App.py ./scenes/sphere.xml
If no command-line argument is provided then renders the default scene file specified in
the variable DEFAULT_SCENE_FILE.

Spyder IDE:
press F6
Choose "Command line options"
In the text field provide the scene file e.g. ./scenes/sphere.xml
Press Run

Getting Started:
----------------
Initially you'd find it easier to execute Scene.py which contains a basic scene.
Once you're ready to try more complex scenes you should use this file. This file
takes the xml scene file as command line argument and calls the SceneParser
to parse the file and finally render the scene contained in the file.
To speed up rendering you can edit the xml file and reduce the image size
by modifying the 'width' and 'height' attribute in the <camera> element.

Note: For testing your implementaion of intersect method you should use the
unit-test files that are provided e.g. TestSphereIntersection.py, TestBoxIntersection.py,
TestSceneNodeIntersection.py, and TestPlaneIntersection.py
'''

import os, sys
from SceneParser import SceneParser


DEFAULT_SCENE_FILE = './scenes/sphere.xml' 

def build_raytracer():
    ''' 
    For building binary using cython (not used by default). Requires
    C compiler such as msvc/gcc/clang
    '''
    compiler_opt = ''
    python_path_prefix = '' #'C:/Anaconda/' # provide path to python if needed
    
    # cythonize using setup.py
    cmd = python_path_prefix + 'python setup.py build_ext --inplace ' + compiler_opt
    os.system(cmd)

def main(filename):
    scene = SceneParser(filename).scene
    ## to disable showing images in matplotlib uncomment the following line
    #scene.render.bShowImage = False
    scene.renderScene()
  
  
if __name__ == '__main__':
    argv = sys.argv
    filename = DEFAULT_SCENE_FILE
    if len(argv) == 2:
        filename = argv[1]
        
    main(filename)
