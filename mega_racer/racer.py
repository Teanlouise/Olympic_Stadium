from OpenGL.GL import *
import glfw
import numpy as np
import math
from PIL import Image
import imgui

import lab_utils as lu
from lab_utils import vec3, vec2
from terrain import TerrainInfo
from ObjModel import ObjModel



class Racer:
    position = vec3(0,0,0)
    velocity = vec3(0,0,0)
    heading = vec3(1,0,0)
    speed = 0.0

    maxSpeedRoad = 50.0
    maxSpeedRough = 15.0
    zOffset = 3.0
    angvel = 2.0

    terrain = None
    model = None

    def render(self, view, renderingSystem):
        # TODO 1.3: This is a good place to draw the racer model instead of the sphere
        #lu.drawSphere(self.position, 1.5, [1,0,0,1], view)
        modelToWorldTransform = lu.make_mat4_from_zAxis(self.position, self.heading, [ 0.0, 0.0, 1.0 ])
        renderingSystem.drawObjModel(self.model, modelToWorldTransform, view)

    def load(self, objModelName, terrain, renderingSystem):
        self.terrain = terrain
        self.position = terrain.startLocations[0]
        # TODO 1.3: This is a good place create and load the racer model
        self.model = ObjModel(objModelName)

    def update(self, dt, keyStateMap):
        info = self.terrain.getInfoAt(self.position);
        # Select max speed based on material
        maxSpeed = self.maxSpeedRoad if info.material == TerrainInfo.M_Road else self.maxSpeedRough;

        targetVel = vec3(0.0)
        if keyStateMap["UP"]:
            targetVel = self.heading * maxSpeed;
        if keyStateMap["DOWN"]:
            targetVel = self.heading * -maxSpeed;
        
        # linearly interpolate towards the target velocity - this means it is tied to the frame rate, which kind of is bad.
        self.velocity = lu.mix(self.velocity, targetVel, 0.01)

        self.speed = lu.length(self.velocity);

        rotationMat = lu.Mat4()
        if keyStateMap["LEFT"]:
            rotationMat = lu.make_rotation_z(dt * self.angvel)
        if keyStateMap["RIGHT"]:
            rotationMat = lu.make_rotation_z(dt * -self.angvel)

        self.heading = lu.Mat3(rotationMat) * self.heading;

        # get height of ground at this point.

        self.position += self.velocity * dt;

        # TODO 1.1: After the terrain height is correct, uncomment this line to make the racer follow the ground height
        self.position[2] = lu.mix(self.position[2], info.height + self.zOffset, 0.1);

    def drawUi(self):
        imgui.label_text("Speed", "%0.1fm/s"%self.speed)

