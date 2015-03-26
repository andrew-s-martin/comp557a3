import numpy as np
import GeomTransform as GT

class Ray:
  """
  
  Ray
  
  The ray class represents a ray casted from the eyePoint in the viewDirection.  
  A line equation can be used to get a point along the ray.
  
  p = viewDirection * t + eyePoint.
  
  This means that it assumes that the viewDirection vector is of length 1 in the
  object's coordinate space. Note that we do not explicitly normalize the 
  viewDirection here. This is because this class's method may get called from
  SceneNode which might apply some transformation to viewDirection and use the
  transformed viewDirection to compute the distance to the point.
  
  getPoint(self, t)
  The method getPoint(t) returns the point according to the above line equation.
  
  """
    
  def __init__(self, eyePoint = [0.0, 0.0, 0.0], viewDirection=[0.0, 0.0, 0.0]):
    self.eyePoint = np.array(eyePoint)
    self.viewDirection = np.array(viewDirection)
  
  def __str__(self):
    return 'eye : ' + str(self.eyePoint) + ' , viewDir : ' + str(self.viewDirection)
    
  def getPoint(self, t):
    return self.eyePoint + self.viewDirection * t


class IntersectionResult:
  """
  
  IntersectionResult
  
  This helper class holds all the information regarding an intersection point along the
  ray.  It is meant to be used to compare different intersection points along a ray by
  looking at the value of t.  Also, it holds the normal of the intersected surface and
  the material, both required to compute the lighting.
  
  """
  ZeroVec3 = np.array([0.0, 0.0, 0.0])
  def __init__(self):
    self.n = self.ZeroVec3
    self.p = self.ZeroVec3
    self.material = None
    self.t = np.inf
    
  def __str__(self):
    return 'n : ' + str(self.n) + ' , p : ' + str(self.p) + ' , t : ' + str(self.t)
    
  def is_valid_intersection(self):
    ''' Any intersection between 1e-9 and inf is considered valid ''' 
    return self.t < np.inf and self.t > 1e-9
  
    
