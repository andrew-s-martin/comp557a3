import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import GeomTransform as GT
import math

class Camera:
  """

  Camera class
  
  This class contains all the variables associated with a camera. This class
  is automatically created by the SceneParser and can be accessed from the Scene
  class which contains the main rendering routine. Look at the __init__ and
  __compute_projection_view_parameters__ to see what variables are available.
  None of the variables should/need to be set for implementing the raytracer.
  These variables are necessary for creating rays from the camera position.

  """

  def __init__(self, params = {}):
      ''' SceneParser takes the camera specification like the following from the xml file
      <camera name="myCamera" from="4 3 10" to="0 0 0" up="0 1 0" fov="45" width="400" height="400" />
      and calls this method with all the required parameters. For this assignment
      you're only required to know what the variable names are.
      '''
      self.name = params.get('name')         
      self.pointFrom = np.array(params.get('from', [0., 0., 8.])) # camera's eye position
      self.pointTo = np.array(params.get('to', [0., 0., 0.])) # point in the scene the camera is looking at
      self.up = np.array(params.get('up', [0., 1., 0.])) # camera's up vector
      self.fov = float(params.get('fov', 45.)) # field of view (same in both direction)
      self.near = float(params.get('near', 1)) # distance to the near plane
      self.imageWidth = int(params.get('width', 400))
      self.imageHeight = int(params.get('height', 400))
      self.imageSize = (self.imageWidth, self.imageHeight)
      #print(self.name, self.pointFrom, self.pointTo, self.up, self.fov, self.imageSize)
      
      # use the above variables to compute the camera's projection and view parameters
      self.__compute_projection_view_parameters__()
      
  def __compute_projection_view_parameters__(self):
      '''
      This function is only called from the Camera class for computing
      the relevant projection-view parameters. The function should only get called
      by the Camera.__init__  method. The variables set here can be used for
      ray construction.
      '''
      self.lookat = GT.normalize(self.pointTo - self.pointFrom)
      self.aspect = float(self.imageWidth)/self.imageHeight
      
      # compute frustum 
      self.top = self.near * math.tan(0.5*math.radians(self.fov))
      self.bottom = -self.top
      self.right = self.top*self.aspect
      self.left = -self.right
      
      # Compute the camera coordinate system
      self.cameraZAxis = self.lookat
      self.cameraXAxis = np.cross(self.cameraZAxis, self.up)
      assert(np.linalg.norm(self.cameraXAxis) > 0.) # we don't want the camera coordinate system to collapse
      self.cameraXAxis = GT.normalize(self.cameraXAxis)
      self.cameraYAxis = GT.normalize(np.cross(self.cameraXAxis,self.cameraZAxis))
    
class Render:
  """

  Render class

  The Render class deals with the displaying and saveing the rendered image. 
  It contains the Camera object, an output file name and a background color.

  The Render object is initialized and passed to the Scene object in the scene
  parser (SceneParser.py). The method setPixel is called in the renderScene
  method and the image is saved at the end.
  
  """
  OUTDIR_REL_PATH = './images/' # relative path to output dir
  def __init__(self, params = {}):
      ''' <render bgcolor="0 0 0" output="boxes.png" samples="4" jitter ="true" eyepoints = "1" lensSize = "0.20"> '''    
      self.done = False
      self.camera = params.get('camera', Camera())
      self.output = params.get('output', 'render.png')
      self.bgcolor = np.array(params.get('bgcolor', [0.0,0.0,0.0]))
      self.samples = int(params.get('samples', 1))
      self.jitter = True if params.get('jitter', 'false').upper() is 'TRUE' else False
      self.eyepoints = int(params.get('eyepoints', 1))
      self.lensSize = float(params.get('lensSize', .2))
      self.bShowImage = True if params.get('show_image', 'true').upper() == 'TRUE' else False
      self.OUTDIR_REL_PATH = params.get('out_dir_rel_path', self.OUTDIR_REL_PATH)
      
      #print(params)
      #print(self.camera, self.output, self.bgcolor, self.samples, self.jitter, self.eyepoints, self.lensSize)
      # create image output directory if it doesn't exist
      if not os.path.exists(self.OUTDIR_REL_PATH): 
        os.makedirs(self.OUTDIR_REL_PATH)
        
  def init(self, width, height):
    self.image = Image.new("RGB",(width, height) ,"black")
    
  def setPixel(self, pixel, color):
    """
    Set the pixel to the value. Here color is considered to be floating-point
    """
    #assert(np.all(color <= 1.0) and np.all(0. <= color))
    ''' PIL's putpixel requires the 2nd argument to be a tuple of unsigned 8 bit integers'''
    #self.image.putpixel(pixel, tuple(np.uint32(color * 255)))
    if type(color) is not np.ndarray: # this shouldn't happen if everything is done correctly
        print(pixel, color)
    self.image.putpixel(pixel, (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
        
  def getPixel(self):
      ''' 
      Generator that returns a pixel. A pixel is a python list with the 
      row and column of the pixel. I.e. pixel = [col, row]
      '''
      for row in range(self.camera.imageSize[1]):
          for col in range(self.camera.imageSize[0]):
              yield [col, row]
      
  def save(self):
    '''
    save the rendered image to a file and also display it using matplotlib
    if bShowImage is set to True.
    '''
    try:
        print('saving image to ' + self.OUTDIR_REL_PATH + self.output + ' ...')
        self.image.save(self.OUTDIR_REL_PATH + self.output, "PNG")
        print('done.')
    except:
        print('Failed to save ' + self.OUTDIR_REL_PATH + self.output)
    
    try:
        if self.bShowImage is True:
            plt.imshow(self.image)
            plt.title('Finished rendering.\nImage stored in ' + self.OUTDIR_REL_PATH + self.output)
            plt.show()
    except:
        print('Unable to display image using matplotlib...check ' + \
                self.OUTDIR_REL_PATH + self.output + ' for result.')
        

class Material:
  """

  Material Class

  Each intersectable holds one or two material. Contains the diffuse and
  specular colors, and the hardness which is related to specular shading.

  Could be extended to allow texture map.

  """
  def __init__(self, params = {}):
      self.name = params.get('name')
      self.diffuse = np.array(params.get('diffuse', [1.0, 0.0, 0.0]))
      self.specular = np.array(params.get('specular', [0.5, 0.5, 0.5]))

      if params.has_key('ambient'):
          self.ambient = np.array(params['ambient'])
      else: # if no ambient component is specified in the xml then use the diffuse component as default ambient value
          self.ambient = self.diffuse
          
      self.hardness = float(params.get('hardness', 50.))
    
class Light:
  """

  Light class

  Holds the position, the power and the color of a light. The Scene object has a
  list of lights, and each of them contribute to the shading of all pixels.

  """
  def __init__(self, params = {}):
      self.name = params.get('name')
      self.color = np.array(params.get('color', [1.0, 1.0, 1.0]))
      self.pointFrom = np.array(params.get('from', [0.0, 5.0, 5.0]))
      self.power = float(params.get('power', 1.))
      self.type = params.get('type', 'point')
  
