# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:49:47 2015

@author: Fahim
"""

''' 
Test Ray-Box Intersection

Test for 8 corners
Test for 6 planes
Test for on one of the planes
Test for on one of the corners
Test shooting ray inside and outside
'''


import unittest
import numpy as np
import numpy.testing as nptest
from Intersectable import Box
from Ray import Ray, IntersectionResult
from TesterCommon import test_intersection_with_result, test_no_intersection
import GeomTransform as GT

class TestBoxAtOriginNoIntersectionWithRay(unittest.TestCase):
    ''' 
    Test basic cases where the ray doesn't intersect the box
    Note: By eye (in the test_eye_* functions) we mean the ray origin.
    By viewing we mean ray direction.
    '''
    def setUp(self):
        self.minPoint = np.array([-.5, -.5, -.5])
        self.maxPoint = -self.minPoint
        self.center = (self.minPoint + self.maxPoint) / 2.
        self.box = Box({'min': self.minPoint, 'max': self.maxPoint})
        #print('center', self.center)
        
    def test_box_creation(self):
        ''' sanity check '''
        nptest.assert_array_equal(self.box.minPoint, self.minPoint)
        nptest.assert_array_equal(self.box.maxPoint, self.maxPoint)
        
    def test_no_intersection_opposite(self):
        ''' intersection at negative distance '''
        origin = [0, 10, 0]
        direction = [0, 1, 0]
        test_no_intersection(self.box, origin, direction)
        
    def test_no_intersection_miss_v0(self):
        origin = [0, 10, 0]
        direction = [1, 0, 0]
        test_no_intersection(self.box, origin, direction)
        
    def test_no_intersection_miss_v1(self):
        origin = [0, 10, 10]
        direction = [5, 0, -10]
        test_no_intersection(self.box, origin, direction)
        
    def test_eye_on_box_top_viewing_up(self):
        boxTopPoint = self.center
        boxTopPoint[1] = self.maxPoint[1]
        origin = boxTopPoint
        direction = [0, 1, 0]
        test_no_intersection(self.box, origin, direction)
        
    def test_eye_on_box_bottom_viewing_down(self):
        boxBottomPoint = self.center
        boxBottomPoint[1] = self.minPoint[1]
        origin = boxBottomPoint
        direction = [0, -1, 0]
        test_no_intersection(self.box, origin, direction)

    def test_eye_on_box_right_viewing_right(self):
        boxPoint = self.center
        boxPoint[0] = self.maxPoint[0]
        origin = boxPoint
        direction = [1, 0, 0]
        test_no_intersection(self.box, origin, direction)

    def test_eye_on_box_left_viewing_left(self):
        boxPoint = self.center
        boxPoint[0] = self.minPoint[0]
        origin = boxPoint
        direction = [-1, 0, 0]
        test_no_intersection(self.box, origin, direction)

    def test_eye_on_box_front_viewing_front(self):
        boxPoint = self.center
        boxPoint[2] = self.maxPoint[2]
        origin = boxPoint
        direction = [0, 0, 1]
        test_no_intersection(self.box, origin, direction)

    def test_eye_on_box_back_viewing_back(self):
        boxPoint = self.center
        boxPoint[2] = self.minPoint[2]
        origin = boxPoint
        direction = [0, 0, -1]
        test_no_intersection(self.box, origin, direction)

    #TODO: test more ray origin on corner
    def test_eye_on_corner_viewing_away(self):
        origin = [0.5, 0.5, 0.5]
        direction = [1, 1, 1]
        test_no_intersection(self.box, origin, direction)
        
        direction = [1, 0, 0]
        test_no_intersection(self.box, origin, direction)
        
        direction = [0, 1, 0]
        test_no_intersection(self.box, origin, direction)
        
        direction = [0, 0, 1]
        test_no_intersection(self.box, origin, direction)
    
class TestBoxAtOriginIntersectionWithRay(unittest.TestCase):
    ''' 
    Test basic cases where the ray intersects the box
    Note: By eye (in the test_eye_* functions) we refer to the ray origin
    '''
    def setUp(self):
        self.minPoint = np.array([-.5, -.5, -.5])
        self.maxPoint = -self.minPoint
        self.center = (self.minPoint + self.maxPoint) / 2.
        self.box = Box({'min': self.minPoint, 'max': self.maxPoint})
        #print('center', self.center)
        
    def test_box_creation(self):
        ''' sanity check '''
        nptest.assert_array_equal(self.box.minPoint, self.minPoint)
        nptest.assert_array_equal(self.box.maxPoint, self.maxPoint)
        
    def test_ray_intersection_top(self):
        ''' intersection at the top of the box '''
        origin = [0, 10, 0]
        direction = [0, -1, 0]
        test_intersection_with_result(self.box, origin, direction, 
                                      isect_pt = [0, 0.5, 0], 
                                      isect_normal = [0, 1., 0], 
                                      isect_dist = 9.5)
        
    def test_ray_intersection_bottom(self):
        ''' intersection at the bottom of the box '''
        origin = [0, -10, 0]
        direction = [0, 1, 0]
        test_intersection_with_result(self.box, origin, direction, 
                                      isect_pt = [0, -0.5, 0], 
                                      isect_normal = [0, -1., 0], 
                                      isect_dist = 9.5)
        
    def test_ray_intersection_right(self):
        origin = [10, 0, 0]
        direction = [-1, 0, 0]
        test_intersection_with_result(self.box, origin, direction, 
                                      isect_pt = [0.5, 0, 0], 
                                      isect_normal = [1, 0., 0], 
                                      isect_dist = 9.5)

    def test_ray_intersection_left(self):
        origin = [-10, 0, 0]
        direction = [1, 0, 0]
        test_intersection_with_result(self.box, origin, direction, 
                                      isect_pt = [-0.5, 0, 0], 
                                      isect_normal = [-1, 0, 0], 
                                      isect_dist = 9.5)
                  
    def test_ray_intersection_front(self):
        origin = [0, 0, 10]
        direction = [0, 0, -1]
        test_intersection_with_result(self.box, origin, direction, 
                                      isect_pt = [0, 0, 0.5], 
                                      isect_normal = [0, 0, 1], 
                                      isect_dist = 9.5)

    def test_ray_intersection_back(self):
        origin = [0, 0, -10]
        direction = [0, 0, 1]
        test_intersection_with_result(self.box, origin, direction, 
                                      isect_pt = [0, 0, -0.5], 
                                      isect_normal = [0, 0, -1], 
                                      isect_dist = 9.5)

    def test_ray_intersection_to_corner_maxpt(self):
        origin = [10, 10, 10]
        direction = [-1, -1, -1]
        expected_pt = [0.5, 0.5, 0.5]
        expected_normal = None
        expected_dist = 10 * np.sqrt(3) - np.linalg.norm(expected_pt)
        test_intersection_with_result(self.box, origin, direction, 
                                      isect_pt = expected_pt, 
                                      isect_normal = expected_normal, 
                                      isect_dist = expected_dist)
                                      
#    def test_ray_intersection_from_corner_maxpt(self):
#        '''ray originates at one corner and goes inside '''
#        origin = self.maxPoint
#        direction = [-1, -1, -1]
#        expected_pt = self.minPoint
#        expected_normal = None
#        expected_dist = np.sqrt(3)
#        test_intersection_with_result(self.box, origin, direction, 
#                                      isect_pt = expected_pt, 
#                                      isect_normal = expected_normal, 
#                                      isect_dist = expected_dist)
#
#    def test_ray_on_front_plane_to_back(self):
#        origin = [0, 0, 0.5]
#        direction = [0, 0, -1]
#        expected_pt = [0, 0, -0.5]
#        expected_normal = [0, 0, -1]
#        expected_dist = 1
#        test_intersection_with_result(self.box, origin, direction, 
#                                      isect_pt = expected_pt, 
#                                      isect_normal = expected_normal, 
#                                      isect_dist = expected_dist)
 
def main(): # to make it easier to import this file and run the tests
    unittest.main()
    
if __name__ == '__main__':
    main()