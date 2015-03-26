# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 09:44:46 2015

@author: Fahim
"""

import numpy as np
import numpy.testing as nptest
from Ray import Ray, IntersectionResult
import GeomTransform as GT

def test_intersection_with_result(obj, origin, direction, isect_pt, isect_normal, isect_dist):
    '''test intersection for a given object and  '''
    ray = Ray(origin, GT.normalize(direction))
    #result = IntersectionResult()
    
    result = obj.intersect(ray)
    #print "p",result.p, isect_pt
    nptest.assert_array_almost_equal(result.p, isect_pt)
    if isect_normal is not None:
        # isect_normal might be set to None when the normal is undefined
        #print "n",result.n, isect_normal
        nptest.assert_array_almost_equal(result.n, isect_normal)
    nptest.assert_almost_equal(result.t, isect_dist)
    
def test_no_intersection(obj, origin, direction):
    test_intersection_with_result(obj, origin, direction, 
                                  [0.,0.,0.], [0.,0.,0.], np.inf)