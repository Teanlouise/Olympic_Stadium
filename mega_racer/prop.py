
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
    treeMax = 50
    treeList = []
    treeModel = None

    rockMax = 20
    rockList = []
    rockModel = None

    allProps = []
  
    def loadPropList(self, propModel, propMax, propLocations):
        propList = []
        i = 0
        while i < propMax:
            prop = Prop()
            prop.load(propModel, propLocations)
            propList.append(prop)            
            i += 1
            print(prop.position)
        return propList

    def loadAllProps(self, terrain):
        # Load trees
        self.treeModel = ObjModel("mega_racer/data/trees/birch_01_d.obj")
        self.treeList = self.loadPropList(self.treeModel, self.treeMax, terrain.treeLocations)
        # Load rocks
        self.rockModel = ObjModel("mega_racer/data/rocks/rock_01.obj")
        self.rockList = self.loadPropList(self.rockModel, self.rockMax, terrain.rockLocations) 
        # Create one large list
        self.allProps = self.treeList + self.rockList

    def renderAllProps(self, view, renderingSystem):
        for prop in self.allProps:
            prop.render(view, renderingSystem)

