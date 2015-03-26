# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 20:48:59 2015

@author: Fahim
"""
from Intersectable import Sphere, Plane, Box, SceneNode
from HelperClasses import Material, Light, Camera, Render
from Scene import Scene
import GeomTransform as GT
import numpy as np
from xml.dom import minidom
import copy

'''
Scene file requirements:
------------------------
+ One scene per file
+ All xml element names are case-sensitive
+ Scene description must be inside <scene></scene>
+ <material> tag contains attributes for Material class
+ <material> tag needs to have a name attribute if they are referred from elsewhere
+ If two elements have the same node then only one of them is used.
+ <light> tag has attributes for Light class
+ <render> tag provides the parameters for Render class
+ <camera> tag goes inside <render> and contains the Camera class parameters
+ Intersectable objects have tags <sphere>, <plane>, <box>, <node>
+ A node (SceneNode) can refer to another node by name using "ref=" E.g. 
<node name="base">...</node>
<node name="ref_example" ref="base"></node>
+ SceneParser process each XML element and generates a dictionary containing
all the parameters of a relevant class. The class's __init__ method takes
the dictionary and does all the necessary processing. 
+ Adding new attribute to an element:
This can be done by simply adding the attribute to the xml file and retrieving
that attribute from the parameter dictionary in the corresponding classes
__init__ file.
'''
class SceneParser:
    ''' 
    Parses an XML file and create a Scene class. The default value of a 
    particular variable will go into the class that uses that variable. 
    This class is implemented so that it doesn't have to know a lot about
    the individual nodes in the scene. For instance light may have some special
    variables that only the Light class knows how to process. In that case
    it makes more sense to just pass all the parameters to Light class for it to
    do the processing. Therefore, this class basically builds all the attributes
    of a node into a parameter dictionary and passes it to the appropriate class.
    This class does however need to know the variable names and datatype and
    the nodes that can be in the scene description file. __var_datatypes store
    the mapping between variable name and their type. New variables and their
    type can simply be added to this dictionary. This information can also be
    stored as an external file and loaded here. Node types can also be generalized
    by modifying the xml file schema but that is not done for consistency with
    the existing xml format (this would require a major rewrite of the xml files.)
    When the parser doesn't recognize a node it simply ignores it.
    '''
    __var_datatypes = {'ambient': np.ndarray, 'color': np.ndarray,
                   'name': str, 'from' : np.ndarray, 'to': np.ndarray,
                   'up': np.ndarray,
                   'radius' : float, 'center' : np.ndarray,
                   'power': float, 'type': str,
                   'diffuse': np.ndarray, 'specular': np.ndarray,
                   'hardness': float,
                   'rotation': np.ndarray, 'translation': np.ndarray,
                   'scale' : np.ndarray,
                   'bgcolor' : np.ndarray,
                   'min': np.ndarray, 'max' : np.ndarray,
                   'normal': np.ndarray, 'coeffs': np.ndarray,
                   }
            
    def __init__(self, filename):
        print('Parsing ' + filename)
        self.Materials = dict() # used for storing ref to materials
        self.Nodes = dict()     # used for storing ref to nodes
        self.NodeStack = []
        self.scene = Scene()
        xml = minidom.parse(filename)
        self.parse(xml)
        print('parsing done.')
        
    def method_dispatcher(self, method_name, arg):
        if(hasattr(self, method_name)):    
            handler = getattr(self, method_name)
            return handler(arg)
        else:
            print('WARNING: %s METHOD NOT FOUND' % method_name)
            
    def parse(self, node):
        method_name = 'parse_' + node.__class__.__name__
        return self.method_dispatcher(method_name, node)
        
    def parse_Document(self, node):
        self.parse(node.documentElement)
    
    def parse_Element(self, node):
        return self.method_dispatcher('process_' + node.tagName, node)
    
    def parse_Text(self, node):
        pass  # ignore Text node
    
    def parse_Comment(self, node):
        pass # ignore Comment node
    
    def convert_attribute_value(self, value, var_type):
        if var_type is float:
            return float(value)
        elif var_type is int:
            return int(value)
        elif var_type is np.ndarray:
            return np.array(map(float, str(value).split()))
        else: # return as string
            return value 
            
    def create_params(self, attribs):
        params = dict()
        for key in attribs.keys():
            var_type = self.__var_datatypes.get(key, str)
            params[key] = self.convert_attribute_value(attribs[key].value, var_type)
        return params
    
    def create_params_from_child(self, node, params):    
        for e in node.childNodes:
            val = self.parse(e)
            if val is not None:
                key = str(val.__class__.__name__).lower()
                if params.has_key(key):
                    print('key ' + key + ' already exists')
                    if type(params[key]) is list:
                        print('appending')
                        params[key].append(val)
                    else: #if the key already exists but is not a list
                        print('creating list and appending')
                        cval = params[key]
                        params[key] = [cval, val]
                else:        
                    params[key] = val
        return params
    
    def process_scene(self, node):
        params = self.create_params(node.attributes)
        self.scene.set_params(params)
        #self.NodeStack.append(SceneNode()) # root scene node
        
        for e in node.childNodes:
            self.parse(e)
        #self.scene.surfaces.append(self.NodeStack.pop())
        
    def process_light(self, node):
        ''' <light name="myLight" color="1 1 1" from="0 0 0 " power="1.0" type="point" /> '''
        self.scene.lights.append(Light(self.create_params(node.attributes)))
        
    def process_material(self, node):
        '''<material name="blue" diffuse="0 0 1" specular="0 0 0" hardness="0" />'''
        params = self.create_params(node.attributes)
        if params.has_key('ref'):
            if not self.Materials.has_key(params['ref']):
                print('Warning: Material ' + params['ref'] + ' not found')
            return self.Materials.get(params['ref'])
        else:
            print('adding Material ' + params['name'])
            material = Material(params)
            self.Materials[params.get('name')] = material
            return material
    
    def process_material2(self, node):
        return self.process_material(node)
        
    def process_node(self, node):
        '''Nodes can refer to other nodes '''
        print('start process_node')
        #self.scene.start_node()
        params = self.create_params(node.attributes)
        #TODO: remove this comment block (currently kept for reference)
#        print(params)
#        rot_angles = np.array(params.get('rotation', [0., 0., 0.]))
#        translate_amount = np.array(params.get('translation', [0., 0., 0.]))
#        scale_amount = np.array(params.get('scale', [1., 1., 1.]))
#        
#        '''The raytracer SceneNode takes care of the hierarchical transformation''' 
#        M = GT.translate(translate_amount) *  GT.rotateX(rot_angles[0]) * \
#            GT.rotateY(rot_angles[1]) * GT.rotateZ(rot_angles[2]) * \
#            GT.scale(scale_amount)
#        print M.getA()
        ''' push the current node before further processing '''
        self.NodeStack.append(SceneNode(params = params))
        if params.has_key('ref'):
            if not self.Nodes.has_key(params['ref']):
                print('WARNING: Ref ' + params['ref'] + ' not found.')
            else: 
                '''handle ref to other nodes. What happens to the root 
                transformation of the ref node? The orig transformation gets
                overridden'''
                print('Found ref node : ' + params['ref'])
                self.NodeStack[-1].children = copy.deepcopy(self.Nodes[params['ref']].children)
                for e in node.childNodes: # what sort of things can be in the childnode of ref nodes?
                    val = self.parse(e)
                    if val is not None: #Handle plane i.e. material2
                        if e.tagName == 'material':
                            self.NodeStack[-1].children[-1].material = val
                        elif e.tagName == 'material2':
                            self.NodeStack[-1].children[-1].material2 = val
        else: # for non ref nodes do the usual processing
            for e in node.childNodes:
                self.parse(e)
            
        topNode = self.NodeStack.pop()
        if len(self.NodeStack) == 0:
            print('adding node to scene')
            self.scene.surfaces.append(topNode)
        else:
            # add to the current node
            self.NodeStack[-1].children.append(topNode)
        # add to node table for future reference
        nodeName = params.get('name')
        if nodeName is not None:
            self.Nodes[nodeName] = topNode
        print('end process_node')
     
    def process_render(self, node):
        params = self.create_params(node.attributes)
        params = self.create_params_from_child(node, params)
        self.scene.render = Render(params)        
        
    def process_camera(self, node):
        camera = Camera(self.create_params(node.attributes))
        return camera
   
    def create_geom_object(self, node, ObjClass):
        params = self.create_params(node.attributes)
        params = self.create_params_from_child(node, params)
        geom_obj = ObjClass(params)
        print('create_geom_object: ' + str(geom_obj))
        # decide if the object is going to be in the SceneNode or appended to the
        # scene directly
        if len(self.NodeStack) == 0:
            self.scene.surfaces.append(geom_obj)
        else:
            self.NodeStack[-1].children.append(geom_obj)
        return geom_obj
        
    def process_sphere(self, node):
        return self.create_geom_object(node, Sphere)
    
    def process_box(self, node):
        return self.create_geom_object(node, Box)
        
    def process_plane(self, node):
        return self.create_geom_object(node, Plane)
        
if __name__ == '__main__':
    SceneParser('./scenes/plane4.xml').scene.renderScene()
    
