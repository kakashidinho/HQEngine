include_directories(${CMAKE_CURRENT_SOURCE_DIR}  
		${OPENGL_INCLUDE_DIRS} 
		../ImagesLoader
		glew/include
		../../../ThirdParty-mod
		 )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGLEW_NO_GLU -DMOJOSHADER_NO_VERSION_INCLUDE")
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -DGLEW_NO_GLU -DMOJOSHADER_NO_VERSION_INCLUDE")

set (MAIN_SRC_FILES 
	HQDeviceEnumGL.cpp HQDeviceGL.cpp HQDeviceGL_caps.cpp HQCommonGL.cpp
	HQRenderTargetFBO.cpp HQShaderGL.cpp  	HQShaderGL_Common.cpp 
	HQShaderGL_GLSLController.cpp 
	HQShaderGL_GLSL_VarParser.cpp HQShaderGL_UBO.cpp  HQShaderGL_FakeUBO.cpp HQShaderGL_ShaderStorageSupported.cpp
	HQFFEmuShaderControllerGL.cpp 
	HQStateManagerGL.cpp 	HQVertexStreamManagerGL.cpp HQVertexStreamManagerGL_VAO.cpp
	HQTextureManagerGL.cpp ../BaseImpl/HQTextureManagerBaseImpl.cpp 
	glew/src/glew.c
)

set (MOJO_SHADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../ThirdParty-mod/mojoshader)

set (MAIN_MOJO_SHADER_SRC_FILES
	${MOJO_SHADER_PATH}/mojoshader_preprocessor.cpp 
	${MOJO_SHADER_PATH}/mojoshader_common.cpp 
	${MOJO_SHADER_PATH}/mojoshader_lexer.cpp 
	${MOJO_SHADER_PATH}/mojoshader.cpp
)


add_library(HQRendererGL  OBJECT
	${MAIN_SRC_FILES}
	${MAIN_MOJO_SHADER_SRC_FILES}
)
