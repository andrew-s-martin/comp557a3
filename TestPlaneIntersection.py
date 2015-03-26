# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:47:25 2015

@author: Fahim
"""


import unittest
import numpy as np
import numpy.testing as nptest
from Intersectable import Plane
from Ray import Ray, IntersectionResult
from TesterCommon import test_intersection_with_result, test_no_intersection

class TestXZPlaneThroughOriginNoIntersectionWithRay(unittest.TestCase):
    ''' Test basic cases where the ray doesn't intersect the plane '''
    def setUp(self):
        self.normal = [0., 1., 0.]
        self.plane = Plane({'normal': self.normal})
        
    def test_plane_creation(self):
        ''' sanity check '''
        nptest.assert_array_equal(self.plane.normal, self.normal)
        
    def test_no_intersection_opposite(self):
        ''' intersection at negative distance '''
        origin = [0, 10, 0]
        direction = [0, 1, 0]
        test_no_intersection(self.plane, origin, direction)
        
    def test_no_intersection_miss_v0(self):
        origin = [0, 10, 0]
        direction = [1, 0, 0]
        test_no_intersection(self.plane, origin, direction)
        
    def test_no_intersection_miss_v1(self):
        origin = [0, 10, 10]
        direction = [5, 0, -10]
        test_no_intersection(self.plane, origin, direction)
        
    def test_eye_on_plane_viewing_up(self):
        origin = [0, 0, 10]
        direction = [0, 1, 0]
        test_no_intersection(self.plane, origin, direction)    
        
    def test_eye_on_plane_viewing_down(self):
        origin = [0, 0, 10]
        direction = [0, -1, 0]
        test_no_intersection(self.plane, origin, direction)
        
    def test_eye_on_plane_viewing_away(self):
        origin = [0, 0, 10]
        direction = [0, 1, -10]
        test_no_intersection(self.plane, origin, direction)
    
    def test_ray_on_plane(self):
        ''' the ray is entirely on the plane'''
        origin = [0, 0, 10]
        direction = [1, 0, -1]
        test_no_intersection(self.plane, origin, direction)
        
class TestXZPlaneThroughOriginIntersectionWithRay(unittest.TestCase):
    ''' Tests intersection with XZ plane '''
    def setUp(self):
        self.normal = [0., 1., 0.]
        self.plane = Plane({'normal': self.normal})
        
    def test_plane_creation(self):
        ''' sanity check '''
        nptest.assert_array_equal(self.plane.normal, self.normal)
        
    def test_parallel_to_normal_view_direction(self):
        eye = [0, 10, 0]
        viewDir = [0, -1, 0]
        test_intersection_with_result(self.plane, origin=eye, 
                                      direction=viewDir,
                                      isect_pt = [0, 0, 0], 
                                      isect_normal=self.normal,
                                      isect_dist = eye[1])
 
def main(): # to make it easier to import this file and run the tests
    unittest.main()
    
if __name__ == '__main__':
    main()