# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:32:08 2015

@author: Fahim
"""

import unittest
import numpy as np
import numpy.testing as nptest
from Intersectable import Plane, Sphere, Box, SceneNode
from Ray import Ray, IntersectionResult
from TesterCommon import test_intersection_with_result, test_no_intersection
import GeomTransform as GT

class TestSceneNodeBasic_LargeSphere(unittest.TestCase):
    ''' 
    Add a sphere of radius 0.5 and scale it up and check if the sphere
    intersection for radius 2 holds.
    '''
    def setUp(self):
        self.scale_factor = 4
        self.scale = np.ones((1, 3)).flatten() * self.scale_factor
        self.center = [0, 0, 0]
        self.radius = 0.5
        self.scaled_radius = self.radius * self.scale_factor
        self.scene_node = SceneNode(params = {'scale': self.scale})
        self.scene_node.children.append(Sphere({'center': self.center, 
                                                'radius': self.radius}))

    def test_scene_node_creation(self):
        ''' sanity check '''
        M = np.eye(4)
        M[:3,:3] = np.diag(self.scale)
        nptest.assert_array_equal(self.scene_node.M, M)
        nptest.assert_array_equal(self.scene_node.Minv, np.linalg.inv(M))
        self.assertEqual(len(self.scene_node.children), 1)
        self.assertEqual(self.scene_node.children[0].__class__.__name__, 'Sphere')
     
    def test_basic_ray_intersection_on_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        test_intersection_with_result(self.scene_node, origin=[0, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = [0, 0, 2], 
                                      isect_normal=[0, 0, 1],
                                      isect_dist = 8.0)
        
    def test_basic_ray_intersection_parallel_to_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        expected_pt = [1, 0, self.scaled_radius * np.sin(np.pi/3)]
        expected_normal = GT.normalize(expected_pt)
        test_intersection_with_result(self.scene_node, origin=[1, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = expected_pt, 
                                      isect_normal= expected_normal,
                                      isect_dist = 10 - expected_pt[2])
                                      
                                      
class TestSceneNodeBasic_Ellipse_X(unittest.TestCase):
    ''' 
    Add an ellipse by scaling a sphere of radius 1 along x by a factor of 10 and check if the ellipse
    intersection for radius 2 holds on y axis.
    '''
    def setUp(self):
        self.scale_factor = 4
        self.scale = np.ones((1, 3)).flatten() * self.scale_factor
        self.scale[0]=40
        self.center = [0, 0, 0]
        self.radius = 0.5
        self.scaled_radius = self.radius * self.scale_factor
        self.scene_node = SceneNode(params = {'scale': self.scale})
        self.scene_node.children.append(Sphere({'center': self.center, 
                                                'radius': self.radius}))

    def test_scene_node_creation(self):
        ''' sanity check '''
        M = np.eye(4)
        M[:3,:3] = np.diag(self.scale)
        nptest.assert_array_equal(self.scene_node.M, M)
        nptest.assert_array_equal(self.scene_node.Minv, np.linalg.inv(M))
        self.assertEqual(len(self.scene_node.children), 1)
        self.assertEqual(self.scene_node.children[0].__class__.__name__, 'Sphere')
     
    def test_basic_ray_intersection_on_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        test_intersection_with_result(self.scene_node, origin=[0, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = [0, 0, 2], 
                                      isect_normal=[0, 0, 1],
                                      isect_dist = 8.0)
        
    def test_basic_ray_intersection_parallel_to_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        expected_pt = [0.0, 2.0, 0.0]
        expected_normal = GT.normalize(expected_pt)
        test_intersection_with_result(self.scene_node, origin=[0, 2, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = expected_pt, 
                                      isect_normal= expected_normal,
                                      isect_dist = 10 - expected_pt[2])                                      

class TestSceneNodeBasic_Ellipse_Y(unittest.TestCase):
    ''' 
    Add an ellipse by scaling a sphere of radius 1 along y by a factor of 10 and check if the ellipse
    intersection for radius 2 holds on y axis.
    '''
    def setUp(self):
        self.scale_factor = 4
        self.scale = np.ones((1, 3)).flatten() * self.scale_factor
        self.scale[1]=40
        self.center = [0, 0, 0]
        self.radius = 0.5
        self.scaled_radius = self.radius * self.scale_factor
        self.scene_node = SceneNode(params = {'scale': self.scale})
        self.scene_node.children.append(Sphere({'center': self.center, 
                                                'radius': self.radius}))

    def test_scene_node_creation(self):
        ''' sanity check '''
        M = np.eye(4)
        M[:3,:3] = np.diag(self.scale)
        nptest.assert_array_equal(self.scene_node.M, M)
        nptest.assert_array_equal(self.scene_node.Minv, np.linalg.inv(M))
        self.assertEqual(len(self.scene_node.children), 1)
        self.assertEqual(self.scene_node.children[0].__class__.__name__, 'Sphere')
     
    def test_basic_ray_intersection_on_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        test_intersection_with_result(self.scene_node, origin=[0, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = [0, 0, 2], 
                                      isect_normal=[0, 0, 1],
                                      isect_dist = 8.0)
        
    def test_basic_ray_intersection_parallel_to_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        expected_pt = [2.0, 0.0, 0.0]
        expected_normal = GT.normalize(expected_pt)
        test_intersection_with_result(self.scene_node, origin=[2, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = expected_pt, 
                                      isect_normal= expected_normal,
                                      isect_dist = 10 - expected_pt[2])
                                      

class TestSceneNodeBasic_Sphere_and_Box(unittest.TestCase):
    ''' 
    Add an ellipse by scaling a sphere of radius 1 along y by a factor of 10 and check if the ellipse
    intersection for radius 2 holds on y axis.
    '''
    def setUp(self):
        self.scale_factor = 4
        self.scale = np.ones((1, 3)).flatten() * self.scale_factor
        self.center = [0, 0, 0]
        self.radius = 0.5
        self.scaled_radius = self.radius * self.scale_factor
        self.scene_node = SceneNode(params = {'scale': self.scale})
        self.scene_node.children.append(Sphere({'center': self.center, 
                                                'radius': self.radius}))
        self.min= [-1,-1,-10]
        self.max= [1,1,-8]
        self.scene_node.children.append(Box({'min': self.min, 
                                                'max': self.max}))

    def test_scene_node_creation(self):
        ''' sanity check '''
        M = np.eye(4)
        M[:3,:3] = np.diag(self.scale)
        nptest.assert_array_equal(self.scene_node.M, M)
        nptest.assert_array_equal(self.scene_node.Minv, np.linalg.inv(M))
        self.assertEqual(len(self.scene_node.children), 2)
        self.assertEqual(self.scene_node.children[0].__class__.__name__, 'Sphere')
        self.assertEqual(self.scene_node.children[1].__class__.__name__, 'Box')
     
    def test_basic_ray_intersection_on_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        test_intersection_with_result(self.scene_node, origin=[0, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = [0, 0, 2], 
                                      isect_normal=[0, 0, 1],
                                      isect_dist = 8.0)
        
    def test_basic_ray_intersection_parallel_to_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        expected_pt = [2.0, 0.0, 0.0]
        expected_normal = GT.normalize(expected_pt)
        test_intersection_with_result(self.scene_node, origin=[2, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = expected_pt, 
                                      isect_normal= expected_normal,
                                      isect_dist = 10 - expected_pt[2])

class TestSceneNodeBasic_Box_and_Sphere(unittest.TestCase):
    ''' 
    Add an ellipse by scaling a sphere of radius 1 along y by a factor of 10 and check if the ellipse
    intersection for radius 2 holds on y axis.
    '''
    def setUp(self):
        self.scale_factor = 4
        self.scale = np.ones((1, 3)).flatten() * self.scale_factor
        self.center = [0, 0, -5]
        self.radius = 0.5
        self.scaled_radius = self.radius * self.scale_factor
        self.min= [-1,-1,-1]
        self.max= [1,1,1]
        self.scene_node = SceneNode(params = {'scale': self.scale})
        self.scene_node.children.append(Box({'min': self.min, 
                                                'max': self.max}))
        self.scene_node.children.append(Sphere({'center': self.center, 
                                                'radius': self.radius}))

    def test_scene_node_creation(self):
        ''' sanity check '''
        M = np.eye(4)
        M[:3,:3] = np.diag(self.scale)
        nptest.assert_array_equal(self.scene_node.M, M)
        nptest.assert_array_equal(self.scene_node.Minv, np.linalg.inv(M))
        self.assertEqual(len(self.scene_node.children), 2)
        self.assertEqual(self.scene_node.children[0].__class__.__name__, 'Box')
        self.assertEqual(self.scene_node.children[1].__class__.__name__, 'Sphere')
     
    def test_basic_ray_intersection_on_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        test_intersection_with_result(self.scene_node, origin=[0, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = [0, 0, 4], 
                                      isect_normal=[0, 0, 1],
                                      isect_dist = 6.0)
        
    def test_basic_ray_intersection_parallel_to_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        expected_pt = [2.0, 0.0, 4.0]
        expected_normal = [0.0,0.0,1.0]
        test_intersection_with_result(self.scene_node, origin=[2, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = expected_pt, 
                                      isect_normal= expected_normal,
                                      isect_dist = 10 - expected_pt[2])


class TestSceneNodeBasic_inside_Box(unittest.TestCase):
    ''' 
    Ray originates from inside a box
    '''
    def setUp(self):
        self.scale_factor = 4
        self.scale = np.ones((1, 3)).flatten() * self.scale_factor
        self.min= [-1,-1,-1]
        self.max= [1,1,1]
        self.scene_node = SceneNode(params = {'scale': self.scale})
        self.scene_node.children.append(Box({'min': self.min, 
                                                'max': self.max}))

    def test_scene_node_creation(self):
        ''' sanity check '''
        M = np.eye(4)
        M[:3,:3] = np.diag(self.scale)
        nptest.assert_array_equal(self.scene_node.M, M)
        nptest.assert_array_equal(self.scene_node.Minv, np.linalg.inv(M))
        self.assertEqual(len(self.scene_node.children), 1)
        self.assertEqual(self.scene_node.children[0].__class__.__name__, 'Box')
     
#==============================================================================
#     def test_basic_ray_intersection_on_zaxis(self):
#         ''' shoot a ray from a point on the z-axis towards the center '''
#         test_intersection_with_result(self.scene_node, origin=[0, 0, 0], 
#                                       direction=[0, 0, -1],
#                                       isect_pt = [0, 0, -4], 
#                                       isect_normal=[0, 0, -1],
#                                       isect_dist = 4.0)
#==============================================================================
        
    def test_basic_ray_intersection_parallel_to_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        expected_pt = [2.0, 0.0, 4.0]
        expected_normal = [0.0,0.0,1.0]
        test_intersection_with_result(self.scene_node, origin=[2, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = expected_pt, 
                                      isect_normal= expected_normal,
                                      isect_dist = 10 - expected_pt[2])

class TestSceneNodeBasic_Plane_from_front(unittest.TestCase):
    ''' 
    Add a plane and create a ray in front of it.
    '''
    def setUp(self):
        self.scale_factor = 4
        self.scale = np.ones((1, 3)).flatten() * self.scale_factor
        self.normal= [0,0,1]
        self.scene_node = SceneNode(params = {'scale': self.scale})
        self.scene_node.children.append(Plane({'normal': self.normal}))

    def test_scene_node_creation(self):
        ''' sanity check '''
        M = np.eye(4)
        M[:3,:3] = np.diag(self.scale)
        nptest.assert_array_equal(self.scene_node.M, M)
        nptest.assert_array_equal(self.scene_node.Minv, np.linalg.inv(M))
        self.assertEqual(len(self.scene_node.children), 1)
        self.assertEqual(self.scene_node.children[0].__class__.__name__, 'Plane')
     
    def test_basic_ray_intersection_on_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        test_intersection_with_result(self.scene_node, origin=[0, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = [0, 0, 0], 
                                      isect_normal=[0, 0, 1],
                                      isect_dist = 10.0)
        
    def test_basic_ray_intersection_parallel_to_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        expected_pt = [2.0, 0.0, 0.0]
        expected_normal = [0.0,0.0,1.0]
        test_intersection_with_result(self.scene_node, origin=[2, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = expected_pt, 
                                      isect_normal= expected_normal,
                                      isect_dist = 10 - expected_pt[2])

                                      
class TestSceneNodeBasic_Plane_from_behind(unittest.TestCase):
    ''' 
    Add a plane and have a ray come from behind it
    '''
    def setUp(self):
        self.scale_factor = 4
        self.scale = np.ones((1, 3)).flatten() * self.scale_factor
        self.normal= [0,0,-1]
        self.scene_node = SceneNode(params = {'scale': self.scale})
        self.scene_node.children.append(Plane({'normal': self.normal}))

    def test_scene_node_creation(self):
        ''' sanity check '''
        M = np.eye(4)
        M[:3,:3] = np.diag(self.scale)
        nptest.assert_array_equal(self.scene_node.M, M)
        nptest.assert_array_equal(self.scene_node.Minv, np.linalg.inv(M))
        self.assertEqual(len(self.scene_node.children), 1)
        self.assertEqual(self.scene_node.children[0].__class__.__name__, 'Plane')
     
    def test_basic_ray_intersection_on_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        test_intersection_with_result(self.scene_node, origin=[0, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = [0, 0, 0], 
                                      isect_normal=[0, 0, -1],
                                      isect_dist = 10.0)
        
    def test_basic_ray_intersection_parallel_to_zaxis(self):
        ''' shoot a ray from a point on the z-axis towards the center '''
        expected_pt = [2.0, 0.0, 0.0]
        expected_normal = [0.0,0.0,-1.0]
        test_intersection_with_result(self.scene_node, origin=[2, 0, 10], 
                                      direction=[0, 0, -1],
                                      isect_pt = expected_pt, 
                                      isect_normal= expected_normal,
                                      isect_dist = 10 - expected_pt[2])                                      
                                      
def main(): # to make it easier to import this file and run the tests
    unittest.main()
    
if __name__ == '__main__':
    main()