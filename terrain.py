from OpenGL.GL import *
import glfw
import numpy as np
import math
from PIL import Image
import imgui

import lab_utils as lu
from lab_utils import vec3, vec2
from ObjModel import ObjModel

# returned by getInfoAt to provide easy access to height and material type on the terrain for use
# by the game logic.
class TerrainInfo:
    M_Road = 0 
    M_Rough = 1
    height = 0.0
    material = 0


# This class looks after loading & generating the terrain geometry as well as rendering.
# It also provides access to the terrain height and type at different points.
class Terrain:
    xyScale = 8.0
    heightScale = 75.0
    textureXyScale = 0.1

    imageWidth = 0
    imageHeight = 0
    shader = None
    renderWireFrame = False

    # Lists of locations generated from the map texture green channel (see the 'load' method)
    # you can add any other meaning of other values as you see fit.
    
    # Locations to place racers at (the start-grid if you will)
    startLocations = []
    # Locations where trees might be put down (the demo implementation samples these randomly)
    treeLocations = []
    # Same for rocks
    rockLocations = []

    # Texture unit allocaitons:
    TU_Grass = 0
    # 2.1
    TU_High = 1
    TU_Steep = 2
    TU_Road = 3
    TU_Map = 4

    # 1.4
    grassTexture = None
    # 2.1
    highTexture = None
    steepTexture = None
    roadTexture = None
    mapTexture = None

    def render(self, view, renderingSystem):
        glUseProgram(self.shader)
        renderingSystem.setCommonUniforms(self.shader, view, lu.Mat4())

        lu.setUniform(self.shader, "terrainHeightScale", self.heightScale);
        lu.setUniform(self.shader, "terrainTextureXyScale", self.textureXyScale);
        xyNormScale = 1.0 / (vec2(self.imageWidth, self.imageHeight) * self.xyScale);
        lu.setUniform(self.shader, "xyNormScale", xyNormScale);
        xyOffset = -(vec2(self.imageWidth, self.imageHeight) + vec2(1.0)) * self.xyScale / 2.0;
        lu.setUniform(self.shader, "xyOffset", xyOffset);

        #TODO 1.4: Bind the grass texture to the right texture unit, hint: lu.bindTexture
        lu.bindTexture(self.TU_Grass, self.grassTexture)
        lu.setUniform(self.shader, "grassTexture", self.TU_Grass)

        # 2.1
        # High
        lu.bindTexture(self.TU_High, self.highTexture)
        lu.setUniform(self.shader, "highTexture", self.TU_High)
        # Steep
        lu.bindTexture(self.TU_Steep, self.steepTexture)
        lu.setUniform(self.shader, "steepTexture", self.TU_Steep)
        # Road
        lu.bindTexture(self.TU_Road, self.roadTexture)
        lu.bindTexture(self.TU_Map, self.mapTexture)
        lu.setUniform(self.shader, "roadTexture", self.TU_Road)
        lu.setUniform(self.shader, "mapTexture", self.TU_Map)


        if self.renderWireFrame:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glLineWidth(1.0);
        glBindVertexArray(self.vertexArrayObject);
        glDrawElements(GL_TRIANGLES, len(self.terrainInds), GL_UNSIGNED_INT, None);

        if self.renderWireFrame:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBindVertexArray(0);
        glUseProgram(0);


    def load(self, imageName, renderingSystem):
        with Image.open(imageName)  as im:
            self.imageWidth = im.size[0]
            self.imageHeight = im.size[1]
            self.imageData = im.tobytes("raw", "RGBX" if im.mode == 'RGB' else "RGBA", 0, -1)

            xyOffset = -vec2(float(self.imageWidth), float(self.imageHeight)) * self.xyScale / 2.0;

            # Calculate vertex positions
            terrainVerts = []
            for j in range(self.imageHeight):
                for i in range(self.imageWidth):
                    offset = (j * self.imageWidth + i) * 4
                    # copy pixel 4 channels
                    imagePixel = self.imageData[offset:offset+4];
                    # Normalize the red channel from [0,255] to [0.0, 1.0]
                    red = float(imagePixel[0]) / 255.0;

                    xyPos = vec2(i, j) * self.xyScale + xyOffset;
                    # TODO 1.1: set the height
                    #zPos = 0.0
                    zPos = red * self.heightScale # Change from zero for 1.1
                    pt = vec3(xyPos[0], xyPos[1], zPos)
                    terrainVerts.append(pt)

                    green = imagePixel[1];
                    if green == 255:
                        self.startLocations.append(pt)
                    if green == 128:
                        self.treeLocations.append(pt)
                    if green == 64:
                        self.rockLocations.append(pt)

            # build vertex normals...
            terrainNormals = [vec3(0.0, 0.0, 1.0)] * self.imageWidth * self.imageHeight;
            for j in range(1, self.imageHeight - 1):
                for i in range(1, self.imageWidth - 1):
                    v = terrainVerts[j * self.imageWidth + i];
                    vxP = terrainVerts[j * self.imageWidth + i - 1];
                    vxN = terrainVerts[j * self.imageWidth + i + 1];
                    dx = vxP - vxN;

                    vyP = terrainVerts[(j - 1) * self.imageWidth + i];
                    vyN = terrainVerts[(j + 1) * self.imageWidth + i];
                    dy = vyP - vyN;

                    nP = lu.normalize(lu.cross(dx, dy));

                    vdxyP = terrainVerts[(j - 1) * self.imageWidth + i - 1];
                    vdxyN = terrainVerts[(j + 1) * self.imageWidth + i + 1];
                    dxy = vdxyP - vdxyN;

                    vdyxP = terrainVerts[(j - 1) * self.imageWidth + i + 1];
                    vdyxN = terrainVerts[(j + 1) * self.imageWidth + i - 1];
                    dyx = vdyxP - vdyxN;

                    nD = lu.normalize(lu.cross(dxy, dyx));

                    terrainNormals[j * self.imageWidth + i] = lu.normalize(nP + nD);



            # join verts with quads that is: 2 triangles @ 3 vertices, with one less in each direction.
            terrainInds = [0] * 2 * 3 * (self.imageWidth - 1) * (self.imageHeight - 1)
            for j in range(0, self.imageHeight - 1):
                for i in range(0, self.imageWidth - 1):
				    # Vertex indices to the four corners of the quad.
                    qInds =[
					    j * self.imageWidth + i,
					    j * self.imageWidth + i + 1,
					    (j + 1) * self.imageWidth + i,
					    (j + 1) * self.imageWidth + i + 1,
				    ]
                    outOffset = 3 * 2 * (j * (self.imageWidth - 1) + i);
                    points = [
					    terrainVerts[qInds[0]],
					    terrainVerts[qInds[1]],
					    terrainVerts[qInds[2]],
					    terrainVerts[qInds[3]],
                    ]
                    # output first triangle:
                    terrainInds[outOffset + 0] = qInds[0];
                    terrainInds[outOffset + 1] = qInds[1];
                    terrainInds[outOffset + 2] = qInds[2];
                    # second triangle
                    terrainInds[outOffset + 3] = qInds[2];
                    terrainInds[outOffset + 4] = qInds[1];
                    terrainInds[outOffset + 5] = qInds[3];
            
            self.terrainInds = terrainInds

            self.vertexArrayObject = lu.createVertexArrayObject();
            self.vertexDataBuffer = lu.createAndAddVertexArrayData(self.vertexArrayObject, terrainVerts, 0);
            self.normalDataBuffer = lu.createAndAddVertexArrayData(self.vertexArrayObject, terrainNormals, 1);
            self.indexDataBuffer = lu.createAndAddIndexArray(self.vertexArrayObject, terrainInds);

            #normalDataBuffer = createAndAddVertexArrayData<vec4>(g_particleVao, { vec4(0.0f) }, 1);



        vertexShader = """
            #version 330
            in vec3 positionIn;
            in vec3 normalIn;

            uniform mat4 modelToClipTransform;
            uniform mat4 modelToViewTransform;
            uniform mat3 modelToViewNormalTransform;

            uniform float terrainHeightScale;
            uniform float terrainTextureXyScale;
            uniform vec2 xyNormScale;
            uniform vec2 xyOffset;


            // 'out' variables declared in a vertex shader can be accessed in the subsequent stages.
            // For a fragment shader the variable is interpolated (the type of interpolation can be modified, try placing 'flat' in front here and in the fragment shader!).
            out VertexData
            {
	            float v2f_height;
                vec3 v2f_viewSpacePosition;
                vec3 v2f_viewSpaceNormal;
                vec3 v2f_worldSpacePosition;
                vec3 v2f_worldSpaceNormal; // 2.1 - Steep

                vec2 v2f_xyNormScale; // 2.1 - Road
                vec2 v2f_xyOffset; // 2.1 - Road

            };

            void main() 
            {
                // pass the world-space Z to the fragment shader, as it is used to compute the colour and other things
	            v2f_height = positionIn.z;
                v2f_worldSpacePosition = positionIn;
                v2f_viewSpacePosition = (modelToViewTransform * vec4(positionIn, 1.0)).xyz;
                v2f_viewSpaceNormal = modelToViewNormalTransform * normalIn;

                v2f_worldSpaceNormal = normalIn; //2.1 - Steep
                v2f_xyNormScale = xyNormScale; // 2.1 - Road
                v2f_xyOffset = xyOffset; // 2.1 - Road

	            // gl_Position is a buit-in 'out'-variable that gets passed on to the clipping and rasterization stages (hardware fixed function).
                // it must be written by the vertex shader in order to produce any drawn geometry. 
                // We transform the position using one matrix multiply from model to clip space. Note the added 1 at the end of the position to make the 3D
                // coordinate homogeneous.
	            gl_Position = modelToClipTransform * vec4(positionIn, 1.0);
            }
"""

        fragmentShader = """
            // Input from the vertex shader, will contain the interpolated (i.e., area weighted average) vaule out put for each of the three vertex shaders that 
            // produced the vertex data for the triangle this fragmet is part of.
            in VertexData
            {
	            float v2f_height;
                vec3 v2f_viewSpacePosition;
                vec3 v2f_viewSpaceNormal;
                vec3 v2f_worldSpacePosition;

                vec3 v2f_worldSpaceNormal;  //2.1 - Steep
                vec2 v2f_xyNormScale; // 2.1 - Road
                vec2 v2f_xyOffset; // 2.1 - Road
            };

            uniform float terrainHeightScale;
            uniform float terrainTextureXyScale;
            // 1.4
            uniform sampler2D grassTexture; 
            // 2.1
            uniform sampler2D highTexture;
            uniform sampler2D steepTexture;   
            uniform sampler2D roadTexture;
            uniform sampler2D mapTexture;

            out vec4 fragmentColor;

            void main() 
            {
                vec3 materialColour = vec3(v2f_height/terrainHeightScale);
                // TODO 1.4: Compute the texture coordinates and sample the texture for the grass and use as material colour.                 
                vec3 grassColour = texture(grassTexture, v2f_worldSpacePosition.xy * terrainTextureXyScale).xyz;
                materialColour = grassColour;

                // 2.1                 
                float slope = dot(v2f_worldSpaceNormal, vec3(v2f_worldSpaceNormal.x, 0.0, v2f_worldSpaceNormal.z)); //Steep 
                float blueChannel = texture(mapTexture, (v2f_worldSpacePosition.xy - v2f_xyOffset) * v2f_xyNormScale).z; //Road

                // Road Texture
                if (blueChannel >= 0.9) {
                    vec3 roadColour = texture(roadTexture, v2f_worldSpacePosition.xy * terrainTextureXyScale).xyz;
                    materialColour = roadColour;
                // Steep texture
                } else if (slope < 0.8) {
                    vec3 steepColour = texture(steepTexture, v2f_worldSpacePosition.xy * terrainTextureXyScale).xyz;
                    materialColour = mix(materialColour, steepColour, (v2f_height/terrainHeightScale));                
                // High texture
                } else if (v2f_height > 60) {
                    vec3 highColour = texture(highTexture, v2f_worldSpacePosition.xy * terrainTextureXyScale).xyz;
                    materialColour = mix(materialColour, highColour, (v2f_height/terrainHeightScale));
                }

                


                vec3 reflectedLight = computeShading(materialColour, v2f_viewSpacePosition, v2f_viewSpaceNormal, viewSpaceLightPosition, sunLightColour);
	            fragmentColor = vec4(toSrgb(reflectedLight), 1.0);
	            //fragmentColor = vec4(toSrgb(vec3(v2f_height/terrainHeightScale)), 1.0);

            }
"""
        # Note how we provide lists of source code strings for the two shader stages.
        # This is basically the only standard way to 'include' or 'import' code into more than one shader. The variable renderingSystem.commonFragmentShaderCode
        # contains code that we wish to use in all the fragment shaders, for example code to transform the colour output to srgb.
        # It is also a nice place to put code to compute lighting and other effects that should be the same accross the terrain and racer for example.
        self.shader = lu.buildShader([vertexShader], ["#version 330\n", renderingSystem.commonFragmentShaderCode, fragmentShader], {"positionIn" : 0, "normalIn" : 1})
        
        # TODO 1.4: Load texture and configure the sampler
        self.grassTexture = ObjModel.loadTexture("grass2.png", "data", True)

        # 2.1
        self.highTexture = ObjModel.loadTexture("rock 2.png", "data", True)
        self.steepTexture = ObjModel.loadTexture("rock 5.png", "data", True)

        self.roadTexture = ObjModel.loadTexture("paving 5.png", "data", True)
        self.mapTexture = ObjModel.loadTexture("track_01_128.png", "data", False)


    # Called by the game to drawt he UI widgets for the terrain.
    def drawUi(self):
        # height scale is read-only as it is not run-time changable (since we use it to compute normals at load-time)
        imgui.label_text("terrainHeightScale", "%0.2f"%self.heightScale)
        #_,self.heightScale = imgui.slider_float("terrainHeightScale", self.heightScale, 1.0, 100.0)
        _,self.textureXyScale = imgui.slider_float("terrainTextureXyScale", self.textureXyScale, 0.01, 10.0)
        _,self.renderWireFrame = imgui.checkbox("WireFrame", self.renderWireFrame);


    # Retrieves information about the terrain at some x/y world-space position, if you request info from outside 
    # the track it just clamps the position to the edge of the track.
    # Returns an instance of the class TerrainInfo.
    # An improvement that would make the movement of the racer smoother is to bi-linearly interpolate the image data.
    # this is something that comes free with OpenGL texture sampling, but here we'd have to implement it ourselves.
    def getInfoAt(self, position):
        # 1. convert x&y to texture scale
        xyOffset = -vec2(self.imageWidth, self.imageHeight) * self.xyScale / 2.0
        imageSpacePos = (vec2(position[0], position[1]) - xyOffset) / self.xyScale;

        x = max(0, min(self.imageWidth - 1, int(imageSpacePos[0])))
        y = max(0, min(self.imageHeight - 1, int(imageSpacePos[1])))
        pixelOffset = (y * self.imageWidth + x) * 4
        # copy pixel 4 channels
        imagePixel = self.imageData[pixelOffset:pixelOffset+4]

        info = TerrainInfo();
        info.height = float(imagePixel[0]) * self.heightScale / 255.0;
        info.material = TerrainInfo.M_Road if imagePixel[2] == 255 else TerrainInfo.M_Rough
        return info;
