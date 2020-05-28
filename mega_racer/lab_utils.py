from OpenGL.GL import *
import numpy as np
import math
from ctypes import sizeof, c_float, c_void_p, c_uint, string_at
import imgui

def vec2(x, y = None):
    if y == None:
        return np.array([x,x], dtype=np.float32)
    return np.array([x,y], dtype=np.float32)

def vec3(x, y = None, z = None):
    if y == None:
        return np.array([x,x,x], dtype=np.float32)
    if z == None:
        return np.array([x,y,y], dtype=np.float32)
    return np.array([x, y, z], dtype=np.float32)


# This is a helper class to provide the ability to use * for matrix/matrix and matrix/vector multiplication.
# It also helps out uploading constants and a few other operations as python does not support overloading functions.
# Note that a vector is just represented as a list on floats, and we rely on numpy to take care of the 
class Mat4:
    matData = None
    # Construct a Mat4 from a python array
    def __init__(self, p = [[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]):
        if isinstance(p, Mat3):
            self.matData = np.matrix(np.identity(4))
            self.matData[:3,:3] = p.matData
        else:
            self.matData = np.matrix(p)

    # overload the multiplication operator to enable sane looking transformation expressions!
    def __mul__(self, other):
        # if it is a list, we let numpy attempt to convert the data
        # we then return it as a list also (the typical use case is 
        # for transforming a vector). Could be made more robust...
        if isinstance(other, list):
            return np.array(self.matData.dot(other).flat, dtype=np.float32)
        # Otherwise we assume it is another Mat4 or something compatible, and just multiply the matrices
        # and return the result as a new Mat4
        return Mat4(self.matData.dot(other.matData))
    
    # Helper to get data as a contiguous array for upload to OpenGL
    def getData(self):
        return np.ascontiguousarray(self.matData, dtype=np.float32)

    # note: returns an inverted copy, does not change the object (for clarity use the global function instead)
    #       only implemented as a member to make it easy to overload based on matrix class (i.e. 3x3 or 4x4)
    def _inverse(self):
        return Mat4(np.linalg.inv(self.matData))

    def _transpose(self):
        return Mat4(self.matData.T)

    def _set_open_gl_uniform(self, loc):
        glUniformMatrix4fv(loc, 1, GL_TRUE, self.getData())



class Mat3:
    matData = None
    # Construct a Mat4 from a python array
    def __init__(self, p = [[1,0,0],
                            [0,1,0],
                            [0,0,1]]):
        if isinstance(p, Mat4):
            self.matData = p.matData[:3,:3]
        else:
            self.matData = np.matrix(p)

    # overload the multiplication operator to enable sane looking transformation expressions!
    def __mul__(self, other):
        # if it is a list, we let numpy attempt to convert the data
        # we then return it as a list also (the typical use case is 
        # for transforming a vector). Could be made more robust...
        if isinstance(other, list) or isinstance(other, np.ndarray):
            return np.array(self.matData.dot(other).flat, dtype=np.float32)
        # Otherwise we assume it is another Mat3 or something compatible, and just multiply the matrices
        # and return the result as a new Mat3
        return Mat3(self.matData.dot(other.matData))
    
    # Helper to get data as a contiguous array for upload to OpenGL
    def getData(self):
        return np.ascontiguousarray(self.matData, dtype=np.float32)

    # note: returns an inverted copy, does not change the object (for clarity use the global function instead)
    #       only implemented as a member to make it easy to overload based on matrix class (i.e. 3x3 or 4x4)
    def _inverse(self):
        return Mat3(np.linalg.inv(self.matData))

    def _transpose(self):
        return Mat3(self.matData.T)

    def _set_open_gl_uniform(self, loc):
        glUniformMatrix3fv(loc, 1, GL_TRUE, self.getData())


#
# matrix consruction functions
#

def make_translation(x, y, z):
    return Mat4([[1,0,0,x],
                 [0,1,0,y],
                 [0,0,1,z],
                 [0,0,0,1]])


def make_translation(x, y, z):
    return Mat4([[1,0,0,x],
                 [0,1,0,y],
                 [0,0,1,z],
                 [0,0,0,1]])

 
def make_scale(x, y, z):
    return Mat4([[x,0,0,0],
                 [0,y,0,0],
                 [0,0,z,0],
                 [0,0,0,1]])


def make_rotation_y(angle):
    return Mat4([[math.cos(angle), 0, math.sin(angle),0],
                 [0,1,0,0],
                 [-math.sin(angle),0, math.cos(angle),0],
                 [0,0,0,1]])


def make_rotation_x(angle):
    return Mat4([[1,0,0,0],
                 [0, math.cos(angle), -math.sin(angle),0],
                 [0, math.sin(angle), math.cos(angle),0],
                 [0,0,0,1]])


def make_rotation_z(angle):
    return Mat4([[math.cos(angle),-math.sin(angle),0,0],
                 [math.sin(angle),math.cos(angle),0,0],
                 [0,0,1,0],
                 [0,0,0,1]])

# 
# This function creates a rotation and translation matrix that aligns the z axis of the matrix with the given direction vector (zAzis)
# very useful for placing objects that have been modelled with the local space z-axis as 'forwards' in the world.
# To form the orthogonal basis, the given yAxis (typically representing 'up' if the z-axis is 'forwards') is used to align the resulting y-aixis
# as far as possible, while ensuring it is perpendicular to the zAxis.
# The 'translation' becomes the translation of the resulting matrix.
# (For some reason glm seems to lack this highly useful functionality)
def make_mat4_from_zAxis(translation, zAxis, yAxis):
    # 1. the z axis is as given.
    z = normalize(zAxis);
    # 2. the x axis is the cross product of the z axis and rough y azis, it is therefore perpendicular to both of these directions.
    x = normalize(cross(yAxis, z));
    # the y axis must be perpendicular to the other two axes, which is constructed using the cross product, (unless they are co-linear).
    y = cross(z, x);

    # There's bound to be a nice and pythonic way to do this...
    return Mat4([[x[0], y[0], z[0], translation[0]],
                 [x[1], y[1], z[1], translation[1]],
                 [x[2], y[2], z[2], translation[2]],
                 [0,   0,   0,    1]])


# 
# Matrix operations
#

# note: returns an inverted copy, does not change the object (for clarity use the global function instead)
def inverse(mat):
    return mat._inverse()

def transpose(mat):
    return mat._transpose()



#
# vector operations
#

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm

def length(v):
    return np.linalg.norm(v)

def cross(a,b):
    return np.cross(a,b)

# Linearly interpolate from v0 to v1, t in [0,1] named to match GLSL
def mix(v0, v1, t):
    return v0 * (1.0 - t) + v1 * t

def dot(a,b):
    return np.dot(a, b)

# The reason we need a 'look from', and don't just use lookAt(pos, pos+dir, up) is because if pos is large (i.e., far from the origin) and 'dir' is a unit vector (common case)
# then the precision loss in the addition followed by subtraction in lookAt to get the direction back is _significant_, and leads to jerky camera movements.
def make_lookFrom(eye, direction, up):
    f = normalize(direction)
    U = np.array(up[:3])
    s = normalize(np.cross(f, U))
    u = np.cross(s, f)
    M = np.matrix(np.identity(4))
    M[:3,:3] = np.vstack([s,u,-f])
    T = make_translation(-eye[0], -eye[1], -eye[2])
    return Mat4(M) * T


# make_lookAt defines a view transform, i.e., from world to view space, using intuitive parameters. location of camera, point to aim, and rough up direction.
# this is basically the same as what we saw in Lexcture #2 for placing the car in the world, except the inverse! (and also view-space 'forwards' is the negative z-axis)
def make_lookAt(eye, target, up):
    return make_lookFrom(eye, np.array(target[:3]) - np.array(eye[:3]), up)



def make_perspective(yFovDeg, aspect, n, f):
    radFovY = math.radians(yFovDeg)
    tanHalfFovY = math.tan(radFovY / 2.0)
    sx = 1.0 / (tanHalfFovY * aspect)
    sy = 1.0 / tanHalfFovY
    zz = -(f + n) / (f - n)
    zw = -(2.0 * f * n) / (f - n)

    return Mat4([[sx,0,0,0],
                 [0,sy,0,0],
                 [0,0,zz,zw],
                 [0,0,-1,0]])



# Turns a multidimensional array (up to 3d?) into a 1D array
def flatten(*lll):
	return [u for ll in lll for l in ll for u in l]


def uploadFloatData(bufferObject, floatData):
    flatData = flatten(floatData)
    data_buffer = (c_float * len(flatData))(*flatData)
    # Upload data to the currently bound GL_ARRAY_BUFFER, note that this is
    # completely anonymous binary data, no type information is retained (we'll
    # supply that later in glVertexAttribPointer)
    glBindBuffer(GL_ARRAY_BUFFER, bufferObject)
    glBufferData(GL_ARRAY_BUFFER, data_buffer, GL_STATIC_DRAW)



def createVertexArrayObject():
	return glGenVertexArrays(1);

def createAndAddVertexArrayData(vertexArrayObject, data, attributeIndex):
    glBindVertexArray(vertexArrayObject)
    buffer = glGenBuffers(1)
    uploadFloatData(buffer, data)

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glVertexAttribPointer(attributeIndex, len(data[0]), GL_FLOAT, GL_FALSE, 0, None);
    glEnableVertexAttribArray(attributeIndex);

    # Unbind the buffers again to avoid unintentianal GL state corruption (this is something that can be rather inconventient to debug)
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return buffer

def createAndAddIndexArray(vertexArrayObject, indexData):
    glBindVertexArray(vertexArrayObject);
    indexBuffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, indexBuffer);

    data_buffer = (c_uint * len(indexData))(*indexData)
    glBufferData(GL_ARRAY_BUFFER, data_buffer, GL_STATIC_DRAW);

    # Bind the index buffer as the element array buffer of the VAO - this causes it to stay bound to this VAO - fairly unobvious.
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);

    # Unbind the buffers again to avoid unintentianal GL state corruption (this is something that can be rather inconventient to debug)
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return indexBuffer;



def getShaderInfoLog(obj):
    logLength = glGetShaderiv(obj, GL_INFO_LOG_LENGTH)

    if logLength > 0:
        return glGetShaderInfoLog(obj).decode()

    return ""



#
# This function performs the steps needed to compile the source code for a
# shader stage (e.g., vertex / fragment) and attach it to a shader program object.
#
def compileAndAttachShader(shaderProgram, shaderType, sources):
    # Create the opengl shader object
    shader = glCreateShader(shaderType)
    # upload the source code for the shader
    # Note the function takes an array of source strings and lengths.
    glShaderSource(shader, sources)
    glCompileShader(shader)

    # If there is a syntax or other compiler error during shader compilation,
    # we'd like to know
    compileOk = glGetShaderiv(shader, GL_COMPILE_STATUS)

    if not compileOk:
        err = getShaderInfoLog(shader)
        print("SHADER COMPILE ERROR: '%s'" % err);
        return False

    glAttachShader(shaderProgram, shader)
    glDeleteShader(shader)
    return True


# creates a shader with a vertex and fragment shader that binds a map of attribute streams 
# to the shader and the also any number of output shader variables
# The fragDataLocs can be left out for programs that don't use multiple render targets as 
# the default for any output variable is zero.
def buildShader(vertexShaderSources, fragmentShaderSources, attribLocs, fragDataLocs = {}):
    shader = glCreateProgram()

    if compileAndAttachShader(shader, GL_VERTEX_SHADER, vertexShaderSources) and compileAndAttachShader(shader, GL_FRAGMENT_SHADER, fragmentShaderSources):
	    # Link the attribute names we used in the vertex shader to the integer index
        for name, loc in attribLocs.items():
            glBindAttribLocation(shader, loc, name)

	    # If we have multiple images bound as render targets, we need to specify which
	    # 'out' variable in the fragment shader goes where in this case it is totally redundant 
        # as we only have one (the default render target, or frame buffer) and the default binding is always zero.
        for name, loc in fragDataLocs.items():
            glBindFragDataLocation(shader, loc, name)

        # once the bindings are done we can link the program stages to get a complete shader pipeline.
        # this can yield errors, for example if the vertex and fragment shaders don't have compatible out and in 
        # variables (e.g., the fragment shader expects some data that the vertex shader is not outputting).
        glLinkProgram(shader)
        linkStatus = glGetProgramiv(shader, GL_LINK_STATUS)
        if not linkStatus:
            err = glGetProgramInfoLog(shader)
            print("SHADER LINKER ERROR: '%s'" % err)
            sys.exit(1)
    return shader




# Helper for debugging, if uniforms appear to not be set properly, you can set a breakpoint here, 
# or uncomment the printing code. If the 'loc' returned is -1, then the variable is either not 
# declared at all in the shader or it is not used  and therefore removed by the optimizing shader compiler.
def getUniformLocationDebug(shaderProgram, name):
    loc = glGetUniformLocation(shaderProgram, name)
    # Useful point for debugging, replace with silencable logging 
    # TODO: should perhaps replace this with the standard python logging facilities
    #if loc == -1:
    #    print("Uniforn '%s' was not found"%name)
    return loc



# Helper to set uniforms of different types, looks the way it does since Python does not have support for 
# function overloading (as C++ has for example). This function covers the types used in the code here, but 
# makes no claim of completeness. The last case is for Mat3/Mat4 (above), and if you get an exception 
# on that line, it is likely because the function was cal
def setUniform(shaderProgram, uniformName, value):
    loc = getUniformLocationDebug(shaderProgram, uniformName)
    if isinstance(value, float):
        glUniform1f(loc, value)
    elif isinstance(value, int):
        glUniform1i(loc, value)
    elif isinstance(value, (np.ndarray, list)):
        if len(value) == 2:
            glUniform2fv(loc, 1, value)
        if len(value) == 3:
            glUniform3fv(loc, 1, value)
        if len(value) == 4:
            glUniform4fv(loc, 1, value)
    elif isinstance(value, (Mat3, Mat4)):
        value._set_open_gl_uniform(loc)
    else:
        assert False # If this happens the type was not supported, check your argument types and either add a new else case above or change the type



# Helper function to extend a 3D point to homogeneous, transform it and back again.
# (For practically all cases except projection, the W component is still 1 after,
# but this covers the correct implementation).
# Note that it does not work for vectors! For vectors we're usually better off just using the 3x3 part of the matrix.
def transformPoint(mat4x4, point):
    x,y,z,w = mat4x4 * [point[0], point[1], point[2], 1.0]
    return vec3(x,y,z) / w


# just a wrapper to convert the returned tuple to a list...
def imguiX_color_edit3_list(label, v):
    a,b = imgui.color_edit3(label, *v)#, imgui.GuiColorEditFlags_Float);// | ImGuiColorEditFlags_HSV);
    return a,list(b)



# Recursively subdivide a triangle with its vertices on the surface of the unit sphere such that the new vertices also are on part of the unit sphere.
def subDivide(dest, v0, v1, v2, level):
	#If the level index/counter is non-zero...
	if level:
		# ...we subdivide the input triangle into four equal sub-triangles
		# The mid points are the half way between to vertices, which is really (v0 + v2) / 2, but 
		# instead we normalize the vertex to 'push' it out to the surface of the unit sphere.
		v3 = normalize(v0 + v1);
		v4 = normalize(v1 + v2);
		v5 = normalize(v2 + v0);

		# ...and then recursively call this function for each of those (with the level decreased by one)
		subDivide(dest, v0, v3, v5, level - 1);
		subDivide(dest, v3, v4, v5, level - 1);
		subDivide(dest, v3, v1, v4, level - 1);
		subDivide(dest, v5, v4, v2, level - 1);
	else:
		# If we have reached the terminating level, just output the vertex position
		dest.append(v0)
		dest.append(v1)
		dest.append(v2)


def createSphere(numSubDivisionLevels):
	sphereVerts = []

	# The root level sphere is formed from 8 triangles in a diamond shape (two pyramids)
	subDivide(sphereVerts, vec3(0, 1, 0), vec3(0, 0, 1), vec3(1, 0, 0), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, 1, 0), vec3(1, 0, 0), vec3(0, 0, -1), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, 1, 0), vec3(0, 0, -1), vec3(-1, 0, 0), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, 1, 0), vec3(-1, 0, 0), vec3(0, 0, 1), numSubDivisionLevels)

	subDivide(sphereVerts, vec3(0, -1, 0), vec3(1, 0, 0), vec3(0, 0, 1), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, -1, 0), vec3(0, 0, 1), vec3(-1, 0, 0), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, -1, 0), vec3(-1, 0, 0), vec3(0, 0, -1), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, -1, 0), vec3(0, 0, -1), vec3(1, 0, 0), numSubDivisionLevels)

	return sphereVerts;


g_sphereVertexArrayObject = None
g_sphereShader = None
g_numSphereVerts = 0

def drawSphere(position, radius, sphereColour, view):
    global g_sphereVertexArrayObject
    global g_sphereShader
    global g_numSphereVerts

    modelToWorldTransform = make_translation(position[0], position[1], position[2]) * make_scale(radius, radius, radius);

    if not g_sphereVertexArrayObject:
        sphereVerts = createSphere(3)
        g_numSphereVerts = len(sphereVerts)
        g_sphereVertexArrayObject = createVertexArrayObject()
        createAndAddVertexArrayData(g_sphereVertexArrayObject, sphereVerts, 0)
        # redundantly add as normals...
        createAndAddVertexArrayData(g_sphereVertexArrayObject, sphereVerts, 1)



        vertexShader = """
            #version 330
            in vec3 positionIn;
            in vec3 normalIn;

            uniform mat4 modelToClipTransform;
            uniform mat4 modelToViewTransform;
            uniform mat3 modelToViewNormalTransform;

            // 'out' variables declared in a vertex shader can be accessed in the subsequent stages.
            // For a fragment shader the variable is interpolated (the type of interpolation can be modified, try placing 'flat' in front here and in the fragment shader!).
            out VertexData
            {
                vec3 v2f_viewSpacePosition;
                vec3 v2f_viewSpaceNormal;
            };

            void main() 
            {
                v2f_viewSpacePosition = (modelToViewTransform * vec4(positionIn, 1.0)).xyz;
                v2f_viewSpaceNormal = normalize(modelToViewNormalTransform * normalIn);

	            // gl_Position is a buit-in 'out'-variable that gets passed on to the clipping and rasterization stages (hardware fixed function).
                // it must be written by the vertex shader in order to produce any drawn geometry. 
                // We transform the position using one matrix multiply from model to clip space. Note the added 1 at the end of the position to make the 3D
                // coordinate homogeneous.
	            gl_Position = modelToClipTransform * vec4(positionIn, 1.0);
            }
"""

        fragmentShader = """
            #version 330
            // Input from the vertex shader, will contain the interpolated (i.e., area weighted average) vaule out put for each of the three vertex shaders that 
            // produced the vertex data for the triangle this fragmet is part of.
            in VertexData
            {
                vec3 v2f_viewSpacePosition;
                vec3 v2f_viewSpaceNormal;
            };

            uniform vec4 sphereColour;

            out vec4 fragmentColor;

            void main() 
            {
                float shading = max(0.0, dot(normalize(-v2f_viewSpacePosition), v2f_viewSpaceNormal));
	            fragmentColor = vec4(sphereColour.xyz * shading, sphereColour.w);

            }
"""
        g_sphereShader = buildShader([vertexShader], [fragmentShader], {"positionIn" : 0, "normalIn" : 1})


    glUseProgram(g_sphereShader)
    setUniform(g_sphereShader, "sphereColour", sphereColour)

    modelToClipTransform = view.viewToClipTransform * view.worldToViewTransform * modelToWorldTransform
    modelToViewTransform = view.worldToViewTransform * modelToWorldTransform
    modelToViewNormalTransform = inverse(transpose(Mat3(modelToViewTransform)))
    setUniform(g_sphereShader, "modelToClipTransform", modelToClipTransform);
    setUniform(g_sphereShader, "modelToViewTransform", modelToViewTransform);
    setUniform(g_sphereShader, "modelToViewNormalTransform", modelToViewNormalTransform);


    glBindVertexArray(g_sphereVertexArrayObject)
    glDrawArrays(GL_TRIANGLES, 0, g_numSphereVerts)


def bindTexture(texUnit, textureId, defaultTexture = None):
	glActiveTexture(GL_TEXTURE0 + texUnit);
	glBindTexture(GL_TEXTURE_2D, textureId if textureId != -1 else defaultTexture);
