from OpenGL.GL import *
import glfw

import numpy as np
from ctypes import sizeof, c_float, c_void_p, c_uint, string_at
import math
import sys
from PIL import Image

import imgui

# we use 'warnings' to remove this warning that ImGui[glfw] gives
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from imgui.integrations.glfw import GlfwRenderer as ImGuiGlfwRenderer

from lab_utils import Mat3, Mat4, make_translation, normalize

from ObjModel import ObjModel
import lab_utils as lu
from lab_utils import vec3, vec2

from terrain import Terrain
from racer import Racer

#
# global variable declarations
#
g_startWidth  = 1280
g_startHeight = 720
g_backGroundColour = [0.1, 0.2, 0.1 ]

g_fov = 60.0
g_nearDistance = 0.2
g_farDistance = 2000.0

g_viewPosition = [ 100.0, 100.0, 100.0 ];
g_viewTarget = [ 0.0, 0.0, 0.0 ];
g_viewUp = [ 0.0, 0.0, 1.0 ];

g_followCamOffset = 25.0
g_followCamLookOffset = 10.0

g_sunStartPosition = [0.0, 0.0, 1000.0]
g_sunPosition = g_sunStartPosition
g_globalAmbientLight = vec3(0.045,0.045,0.045)
g_sunLightColour = [0.9, 0.8, 0.7]

g_updateSun = True
g_sunAngle = 0.0

g_terrain = None
g_racer = None

#
# Key-frames for the sun light and ambient, picked by hand-waving to look ok. Note how most of this is nonsense from a physical point of view and 
# some of the reasoning is basically to compensate for the lack of exposure (or tone-mapping).
#
g_sunKeyFrames = [
	[ -1.0, vec3(0.0, 0.0, 0.0) ], # midnight - no direct light, but we'll ramp up the ambient to make things look ok - should _really_ be done using HDR
	[  0.0, vec3(0.0, 0.0, 0.0) ], # we want ull dark past the horizon line (this ixes shadow artiacts also)
	[  0.2, vec3(0.9, 0.3, 0.2) ], # Quick sunrise to some reddish color perhaps?
	[  1.0, vec3(1.0, 0.9, 0.8) ], # Noon
]
g_ambientKeyFrames = [
	[ -1.0, vec3(0.045, 0.045, 0.15) ], # Night, a somewhat brighter ambient (to compensate or the lack o direct light, totally unphyscal!)
	[  0.0, vec3(0.045, 0.045, 0.15) ], # -||-
	[  0.2, vec3(0.05, 0.045, 0.01) ],  # Quick sunrise to some reddish color perhaps?
	[  1.0, vec3(0.045, 0.045, 0.2) ],  # Noon
]

#
# Classes
#


# ViewParams is used to contain the needed data to represent a 'view' mostly we want to get at the
# projection and world-to-view transform, as these are used in all shaders. Keeping them in one 
# object makes it easier to pass around. It is also convenient future-proofing if we want to add more
# views (e.g., for a shadow map).
class ViewParams:
	viewToClipTransform = lu.Mat4()
	worldToViewTransform = lu.Mat4()
	width = 0
	height = 0



#
# Really just a helper class to be able to pass around shared utilities to the different modules
#
class RenderingSystem:
    # The variable renderingSystem.commonFragmentShaderCode contains code that we wish to use in all the fragment shaders, 
    # for example code to transform the colour output to srgb. It is also a nice place to put code to compute lighting 
    # and other effects that should be the same accross the terrain and racer for example.
    commonFragmentShaderCode = """

    uniform mat4 worldToViewTransform;
    uniform mat4 viewSpaceToSmTextureSpace;
    uniform sampler2DShadow shadowMapTexture;

    uniform vec3 viewSpaceLightPosition;
    uniform vec3 sunLightColour;
    uniform vec3 globalAmbientLight;

    vec3 toSrgb(vec3 color)
    {
      return pow(color, vec3(1.0 / 2.2));
    }


    vec3 computeShading(vec3 materialColour, vec3 viewSpacePosition, vec3 viewSpaceNormal, vec3 viewSpaceLightPos, vec3 lightColour)
    {
        // TODO 1.5: Here's where code to compute shading would be placed most conveniently
        // Direction towards source of light
        vec3 viewSpaceDirToLight = normalize(viewSpaceLightPos - viewSpacePosition);
        // Incoming light intensity      
        float incomingIntensity = max(0.0, dot(viewSpaceNormal, viewSpaceDirToLight));
        // Correct colour and itnensity
        vec3 incomingLight = incomingIntensity * lightColour;
        // Indirect light
        vec3 outgoingLight = (incomingLight + globalAmbientLight) * materialColour;
        return outgoingLight;
        //return materialColour;
        
    }
    """
    objModelShader = None
    # Helper to set common uniforms, such as those used for global lights that should be implemetned the same way in all shaders.
    # Properly, these should be put into a uniform buffer object and uploaded once and for all, this is infinitely faster and better
    # 
    def setCommonUniforms(self, shader, view, modelToWorldTransform):
        # Concatenate the transformations to take vertices directly from model space to clip space
        modelToClipTransform = view.viewToClipTransform * view.worldToViewTransform * modelToWorldTransform
        # Transform to view space from model space (used for the shading)
        modelToViewTransform = view.worldToViewTransform * modelToWorldTransform
        # Transform to view space for normals, need to use the inverse transpose unless only rigid body & uniform scale.
        modelToViewNormalTransform = lu.inverse(lu.transpose(lu.Mat3(modelToViewTransform)))

        # Set the standard transforms, these vary per object and must be set each time an object is drawn (since they have different modelToWorld transforms)
        lu.setUniform(shader, "modelToClipTransform", modelToClipTransform);
        lu.setUniform(shader, "modelToViewTransform", modelToViewTransform);
        lu.setUniform(shader, "modelToViewNormalTransform", modelToViewNormalTransform);
        
        # These transforms are the same for the current view and could be set once for all the objects
        lu.setUniform(shader, "worldToViewTransform", view.worldToViewTransform);
        lu.setUniform(shader, "viewToClipTransform", view.viewToClipTransform);
        # Lighting parameters as could these
        viewSpaceLightPosition = lu.transformPoint(view.worldToViewTransform, g_sunPosition);
        lu.setUniform(shader, "viewSpaceLightPosition", viewSpaceLightPosition);
        lu.setUniform(shader, "globalAmbientLight", g_globalAmbientLight);
        lu.setUniform(shader, "sunLightColour", g_sunLightColour);

    def drawObjModel(self, model, modelToWorldTransform, view):    
        # Bind the shader program such that we can set the uniforms (model.render sets it again)
        glUseProgram(self.objModelShader)

        self.setCommonUniforms(self.objModelShader, view, modelToWorldTransform);
   
        model.render(self.objModelShader)

        glUseProgram(0);        
    
    def setupObjModelShader(self):
        self.objModelShader = lu.buildShader(["""
                #version 330

                in vec3 positionAttribute;
                in vec3	normalAttribute;
                in vec2	texCoordAttribute;

                uniform mat4 modelToClipTransform;
                uniform mat4 modelToViewTransform;
                uniform mat3 modelToViewNormalTransform;

                // Out variables decalred in a vertex shader can be accessed in the subsequent stages.
                // For a pixel shader the variable is interpolated (the type of interpolation can be modified, try placing 'flat' in front, and also in the fragment shader!).
                out VertexData
                {
	                vec3 v2f_viewSpaceNormal;
	                vec3 v2f_viewSpacePosition;
	                vec2 v2f_texCoord;
                };

                void main() 
                {
	                // gl_Position is a buit in out variable that gets passed on to the clipping and rasterization stages.
                  // it must be written in order to produce any drawn geometry. 
                  // We transform the position using one matrix multiply from model to clip space, note the added 1 at the end of the position.
	                gl_Position = modelToClipTransform * vec4(positionAttribute, 1.0);
	                // We transform the normal to view space using the normal transform (which is the inverse-transpose of the rotation part of the modelToViewTransform)
                  // Just using the rotation is only valid if the matrix contains only rotation and uniform scaling.
	                v2f_viewSpaceNormal = normalize(modelToViewNormalTransform * normalAttribute);
	                v2f_viewSpacePosition = (modelToViewTransform * vec4(positionAttribute, 1.0)).xyz;
	                // The texture coordinate is just passed through
	                v2f_texCoord = texCoordAttribute;
                }
                """], ["#version 330\n", self.commonFragmentShaderCode, """
                // Input from the vertex shader, will contain the interpolated (i.e., area-weighted average) vaule out put for each of the three vertex shaders that 
                // produced the vertex data for the triangle this fragmet is part of.
                in VertexData
                {
	                vec3 v2f_viewSpaceNormal;
	                vec3 v2f_viewSpacePosition;
	                vec2 v2f_texCoord;
                };

                // Material properties set by OBJModel.
                uniform vec3 material_diffuse_color; 
	            uniform float material_alpha;
                uniform vec3 material_specular_color; 
                uniform vec3 material_emissive_color; 
                uniform float material_specular_exponent;

                // Textures set by OBJModel 
                uniform sampler2D diffuse_texture;
                uniform sampler2D opacity_texture;
                uniform sampler2D specular_texture;
                uniform sampler2D normal_texture;

                out vec4 fragmentColor;

                void main() 
                {
	                // Manual alpha test (note: alpha test is no longer part of Opengl 3.3).
	                if (texture(opacity_texture, v2f_texCoord).r < 0.5)
	                {
		                discard;
	                }

	                vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;

                    vec3 reflectedLight = computeShading(materialDiffuse, v2f_viewSpacePosition, v2f_viewSpaceNormal, viewSpaceLightPosition, sunLightColour) + material_emissive_color;

	                fragmentColor = vec4(toSrgb(reflectedLight), material_alpha);
                }
            """], ObjModel.getDefaultAttributeBindings())
        glUseProgram(self.objModelShader)
        ObjModel.setDefaultUniformBindings(self.objModelShader)
        glUseProgram(0)





#
# Functions and procedures
#


def sampleKeyFrames(t, kfs):
    # 1. find correct interval
    if t <= kfs[0][0]:
        return kfs[0][1]

    if t >= kfs[-1][0]:
        return kfs[-1][1]

    for i1 in range(1, len(kfs)):
        if t < kfs[i1][0]:
            i0 = i1 - 1;
            t0 = kfs[i0][0];
            t1 = kfs[i1][0];
            # linear interpolation from one to the other
            return lu.mix(kfs[i0][1], kfs[i1][1], (t - t0) / (t1 - t0));

    # we should not get here, unless the key values are malformed (i.e., not strictly increasing)
    assert False
    # but if we do, we return a value that is obviously no good (rahter than say zero that might pass unnoticed for much longer)
    return None




def update(dt, keyStateMap, mouseDelta):
    global g_sunPosition
    global g_sunAngle
    global g_globalAmbientLight 
    global g_sunLightColour
    global g_sunAngle
    global g_updateSun

    global g_viewTarget # Camera Direction
    global g_viewPosition # Camera Position
    global g_followCamOffset # represent the distance behind and above that the camera should be placed
    global g_followCamLookOffset #  used to offset the look target above the racer

    if g_updateSun:
        g_sunAngle += dt * 0.25
        g_sunAngle = g_sunAngle % (2.0 * math.pi);

    g_sunPosition = lu.Mat3(lu.make_rotation_x(g_sunAngle)) * g_sunStartPosition

    g_sunLightColour = sampleKeyFrames(lu.dot(lu.normalize(g_sunPosition), vec3(0.0, 0.0, 1.0)), g_sunKeyFrames)
    g_globalAmbientLight = sampleKeyFrames(lu.dot(lu.normalize(g_sunPosition), vec3(0.0, 0.0, 1.0)), g_ambientKeyFrames)

    g_racer.update(dt, keyStateMap)

    # TODO 1.2: Make the camera look at the racer. Code for updating the camera should be done after the 
    # racer, otherwise the offset will lag and it generally looks weird. 
    
    # Set target position
    g_viewTarget = g_racer.position

    ### Set view position
    for i in range(0,3):
        g_viewPosition[i] = g_racer.position[i] + g_followCamOffset * -g_racer.heading[i]
    g_viewPosition[2] += g_followCamLookOffset


    if imgui.tree_node("Camera", imgui.TREE_NODE_DEFAULT_OPEN):
        _,g_followCamOffset = imgui.slider_float("FollowCamOffset ", g_followCamOffset, 2.0, 100.0)
        _,g_followCamLookOffset = imgui.slider_float("FollowCamLookOffset", g_followCamLookOffset, 0.0, 100.0)
        imgui.tree_pop()

    if imgui.tree_node("Racer", imgui.TREE_NODE_DEFAULT_OPEN):
        g_racer.drawUi()
        imgui.tree_pop()

    if imgui.tree_node("Terrain", imgui.TREE_NODE_DEFAULT_OPEN):
        g_terrain.drawUi()
        imgui.tree_pop()

    if imgui.tree_node("Lighting", imgui.TREE_NODE_DEFAULT_OPEN):
        _,g_globalAmbientLight = lu.imguiX_color_edit3_list("GlobalAmbientLight", g_globalAmbientLight)#, imgui.GuiColorEditFlags_Float);// | ImGuiColorEditFlags_HSV);
        _,g_sunLightColour = lu.imguiX_color_edit3_list("SunLightColour", g_sunLightColour)#, imgui.GuiColorEditFlags_Float);// | ImGuiColorEditFlags_HSV);
        _,g_sunAngle = imgui.slider_float("SunAngle", g_sunAngle, 0.0, 2.0 * math.pi)
        _,g_updateSun = imgui.checkbox("UpdateSun", g_updateSun);
        imgui.tree_pop()




# Called once per frame by the main loop below
def renderFrame(width, height):
    glViewport(0, 0, width, height);
    glClearColor(g_backGroundColour[0], g_backGroundColour[1], g_backGroundColour[2], 1.0)
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

    aspectRatio = float(width) / float(height)

    # ViewParams is used to contain the needed data to represent a 'view' mostly we want to get at the
    # projection and world-to-view transform, as these are used in all shaders. Keeping them in one 
    # object makes it easier to pass around. It is also convenient future-proofing if we want to add more
    # views (e.g., for a shadow map).
    view = ViewParams()
    # Projection (view to clip space transform)
    view.viewToClipTransform = lu.make_perspective(g_fov, aspectRatio, g_nearDistance, g_farDistance)
    # Transform from world space to view space.
    view.worldToViewTransform = lu.make_lookAt(g_viewPosition, g_viewTarget, g_viewUp)
    view.width = width
    view.height = height

    # Call each part of the scene to render itself
    g_terrain.render(view, g_renderingSystem)
    g_racer.render(view, g_renderingSystem)






#
# Main program
#

# String mappings of glfw mouse and keyboard buttons
g_glfwMouseMap = {
    "MOUSE_BUTTON_LEFT" : glfw.MOUSE_BUTTON_LEFT,
    "MOUSE_BUTTON_RIGHT" : glfw.MOUSE_BUTTON_RIGHT,
    "MOUSE_BUTTON_MIDDLE" : glfw.MOUSE_BUTTON_MIDDLE,
}

g_glfwKeymap = {
    "SPACE" : glfw.KEY_SPACE,
    "APOSTROPHE" : glfw.KEY_APOSTROPHE,
    "COMMA" : glfw.KEY_COMMA,
    "MINUS" : glfw.KEY_MINUS,
    "PERIOD" : glfw.KEY_PERIOD,
    "SLASH" : glfw.KEY_SLASH,
    "0" : glfw.KEY_0,
    "1" : glfw.KEY_1,
    "2" : glfw.KEY_2,
    "3" : glfw.KEY_3,
    "4" : glfw.KEY_4,
    "5" : glfw.KEY_5,
    "6" : glfw.KEY_6,
    "7" : glfw.KEY_7,
    "8" : glfw.KEY_8,
    "9" : glfw.KEY_9,
    "SEMICOLON" : glfw.KEY_SEMICOLON,
    "EQUAL" : glfw.KEY_EQUAL,
    "A" : glfw.KEY_A,
    "B" : glfw.KEY_B,
    "C" : glfw.KEY_C,
    "D" : glfw.KEY_D,
    "E" : glfw.KEY_E,
    "F" : glfw.KEY_F,
    "G" : glfw.KEY_G,
    "H" : glfw.KEY_H,
    "I" : glfw.KEY_I,
    "J" : glfw.KEY_J,
    "K" : glfw.KEY_K,
    "L" : glfw.KEY_L,
    "M" : glfw.KEY_M,
    "N" : glfw.KEY_N,
    "O" : glfw.KEY_O,
    "P" : glfw.KEY_P,
    "Q" : glfw.KEY_Q,
    "R" : glfw.KEY_R,
    "S" : glfw.KEY_S,
    "T" : glfw.KEY_T,
    "U" : glfw.KEY_U,
    "V" : glfw.KEY_V,
    "W" : glfw.KEY_W,
    "X" : glfw.KEY_X,
    "Y" : glfw.KEY_Y,
    "Z" : glfw.KEY_Z,
    "LEFT_BRACKET" : glfw.KEY_LEFT_BRACKET,
    "BACKSLASH" : glfw.KEY_BACKSLASH,
    "RIGHT_BRACKET" : glfw.KEY_RIGHT_BRACKET,
    "GRAVE_ACCENT" : glfw.KEY_GRAVE_ACCENT,
    "WORLD_1" : glfw.KEY_WORLD_1,
    "WORLD_2" : glfw.KEY_WORLD_2,
    "ESCAPE" : glfw.KEY_ESCAPE,
    "ENTER" : glfw.KEY_ENTER,
    "TAB" : glfw.KEY_TAB,
    "BACKSPACE" : glfw.KEY_BACKSPACE,
    "INSERT" : glfw.KEY_INSERT,
    "DELETE" : glfw.KEY_DELETE,
    "RIGHT" : glfw.KEY_RIGHT,
    "LEFT" : glfw.KEY_LEFT,
    "DOWN" : glfw.KEY_DOWN,
    "UP" : glfw.KEY_UP,
    "PAGE_UP" : glfw.KEY_PAGE_UP,
    "PAGE_DOWN" : glfw.KEY_PAGE_DOWN,
    "HOME" : glfw.KEY_HOME,
    "END" : glfw.KEY_END,
    "CAPS_LOCK" : glfw.KEY_CAPS_LOCK,
    "SCROLL_LOCK" : glfw.KEY_SCROLL_LOCK,
    "NUM_LOCK" : glfw.KEY_NUM_LOCK,
    "PRINT_SCREEN" : glfw.KEY_PRINT_SCREEN,
    "PAUSE" : glfw.KEY_PAUSE,
    "F1" : glfw.KEY_F1,
    "F2" : glfw.KEY_F2,
    "F3" : glfw.KEY_F3,
    "F4" : glfw.KEY_F4,
    "F5" : glfw.KEY_F5,
    "F6" : glfw.KEY_F6,
    "F7" : glfw.KEY_F7,
    "F8" : glfw.KEY_F8,
    "F9" : glfw.KEY_F9,
    "F10" : glfw.KEY_F10,
    "F11" : glfw.KEY_F11,
    "F12" : glfw.KEY_F12,
    "F13" : glfw.KEY_F13,
    "F14" : glfw.KEY_F14,
    "F15" : glfw.KEY_F15,
    "F16" : glfw.KEY_F16,
    "F17" : glfw.KEY_F17,
    "F18" : glfw.KEY_F18,
    "F19" : glfw.KEY_F19,
    "F20" : glfw.KEY_F20,
    "F21" : glfw.KEY_F21,
    "F22" : glfw.KEY_F22,
    "F23" : glfw.KEY_F23,
    "F24" : glfw.KEY_F24,
    "F25" : glfw.KEY_F25,
    "KP_0" : glfw.KEY_KP_0,
    "KP_1" : glfw.KEY_KP_1,
    "KP_2" : glfw.KEY_KP_2,
    "KP_3" : glfw.KEY_KP_3,
    "KP_4" : glfw.KEY_KP_4,
    "KP_5" : glfw.KEY_KP_5,
    "KP_6" : glfw.KEY_KP_6,
    "KP_7" : glfw.KEY_KP_7,
    "KP_8" : glfw.KEY_KP_8,
    "KP_9" : glfw.KEY_KP_9,
    "KP_DECIMAL" : glfw.KEY_KP_DECIMAL,
    "KP_DIVIDE" : glfw.KEY_KP_DIVIDE,
    "KP_MULTIPLY" : glfw.KEY_KP_MULTIPLY,
    "KP_SUBTRACT" : glfw.KEY_KP_SUBTRACT,
    "KP_ADD" : glfw.KEY_KP_ADD,
    "KP_ENTER" : glfw.KEY_KP_ENTER,
    "KP_EQUAL" : glfw.KEY_KP_EQUAL,
    "LEFT_SHIFT" : glfw.KEY_LEFT_SHIFT,
    "LEFT_CONTROL" : glfw.KEY_LEFT_CONTROL,
    "LEFT_ALT" : glfw.KEY_LEFT_ALT,
    "LEFT_SUPER" : glfw.KEY_LEFT_SUPER,
    "RIGHT_SHIFT" : glfw.KEY_RIGHT_SHIFT,
    "RIGHT_CONTROL" : glfw.KEY_RIGHT_CONTROL,
    "RIGHT_ALT" : glfw.KEY_RIGHT_ALT,
    "RIGHT_SUPER" : glfw.KEY_RIGHT_SUPER,
    "MENU" : glfw.KEY_MENU,
}


if not glfw.init():
    sys.exit(1)


#glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, 1)
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

# This can be used to turn on multisampling AA for the default render target.
#glfw.window_hint(glfw.SAMPLES, g_currentMsaaSamples)


window = glfw.create_window(g_startWidth, g_startHeight, "The Mega-racer game", None, None)
if not window:
    glfw.terminate()
    sys.exit(1)

glfw.make_context_current(window)

print("--------------------------------------\nOpenGL\n  Vendor: %s\n  Renderer: %s\n  Version: %s\n--------------------------------------\n" % (glGetString(GL_VENDOR).decode("utf8"), glGetString(GL_RENDERER).decode("utf8"), glGetString(GL_VERSION).decode("utf8")), flush=True)

# Enable some much-needed hardware functions that are off by default.
glEnable(GL_CULL_FACE);
glEnable(GL_DEPTH_TEST);

impl = ImGuiGlfwRenderer(window)

g_renderingSystem = RenderingSystem()
g_renderingSystem.setupObjModelShader()

g_terrain = Terrain()
#g_terrain.load("data/track_01_32.png", g_renderingSystem);
g_terrain.load("data/track_01_128.png", g_renderingSystem);

g_racer = Racer()
g_racer.load("data/racer_02.obj", g_terrain, g_renderingSystem);

currentTime = glfw.get_time()
prevMouseX,prevMouseY = glfw.get_cursor_pos(window)

while not glfw.window_should_close(window):
    prevTime = currentTime
    currentTime = glfw.get_time()
    dt = currentTime - prevTime

    keyStateMap = {}
    for name,id in g_glfwKeymap.items():
        keyStateMap[name] = glfw.get_key(window, id) == glfw.PRESS

    for name,id in g_glfwMouseMap.items():
        keyStateMap[name] = glfw.get_mouse_button(window, id) == glfw.PRESS


    imgui.new_frame()
    imgui.set_next_window_size(430.0, 450.0, imgui.FIRST_USE_EVER)
    imgui.begin("Tweak variables");

    mouseX,mouseY = glfw.get_cursor_pos(window)
    g_mousePos = [mouseX,mouseY]
    mouseDelta = [mouseX - prevMouseX,mouseY - prevMouseY]
    prevMouseX,prevMouseY = mouseX,mouseY

    # Udpate 'game logic'
    imIo = imgui.get_io()
    if imIo.want_capture_mouse:
        mouseDelta = [0,0]
    update(dt, keyStateMap, mouseDelta)

    width, height = glfw.get_framebuffer_size(window)

    renderFrame(width, height)
    
    #mgui.show_test_window()

    imgui.end()
    imgui.render()
    # Swap front and back buffers
    glfw.swap_buffers(window)

    # Poll for and process events
    glfw.poll_events()
    impl.process_inputs()

glfw.destroy_window(window)
glfw.terminate()
