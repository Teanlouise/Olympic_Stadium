
# Copy from racer.py
from OpenGL.GL import *
#import glfw
#import numpy as np
#import math
#from PIL import Image
#import imgui

import lab_utils as lu
from lab_utils import vec3, vec2
#from terrain import TerrainInfo
from ObjModel import ObjModel

# Import random
import random

class Prop:
    position = vec3(0,0,0)    
    heading = vec3(1,0,0)
    rotation = 0.0

    zOffset = 3.0
    angvel = 2.0
    
    model = None

    def render(self, view, renderingSystem):
        modelToWorldTransform = lu.make_mat4_from_zAxis(self.position, self.heading, [ 0.0, 0.0, 1.0 ])
        rotationTransform = lu.make_rotation_y(self.rotation)
        renderingSystem.drawObjModel(self.model, modelToWorldTransform * rotationTransform, view)

    def load(self, model, locations):      
        self.position = random.choice(locations)
        self.rotation =random.choice(range(0,360))
        self.model = model

class PropManager: 
    propTypes = []
    allProps = []    

    def loadProp(self, propType):
        propModel = ObjModel("data/{propName}/{propName}.obj".format(propName=propType[0]))
        i = 0
        while i < propType[1]:
            prop = Prop()
            prop.load(propModel, propType[2])         
            i += 1
            self.allProps.append(prop)

    def loadAllProps(self, propTypes):
        self.propTypes = propTypes
        for prop in self.propTypes:
            self.loadProp(prop)

    def renderAllProps(self, view, renderingSystem):
        for prop in self.allProps:
            prop.render(view, renderingSystem)

