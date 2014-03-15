include_directories(${CMAKE_CURRENT_SOURCE_DIR}  
		${OPENGL_INCLUDE_DIRS} 
		../ImagesLoader
		glew/include
		../../../ThirdParty-mod
		 )

set (MAIN_SRC_FILES 
	HQDeviceEnumGL.cpp HQDeviceGL.cpp HQDeviceGL_caps.cpp
	HQRenderTargetFBO.cpp HQShaderGL.cpp  	HQShaderGL_Common.cpp 
	HQShaderGL_CgController.cpp HQShaderGL_CombineController.cpp HQShaderGL_GLSLController.cpp 
	HQShaderGL_GLSL_VarParser.cpp HQShaderGL_UBO.cpp
	HQFFEmuShaderControllerGL.cpp 
	HQStateManagerGL.cpp 	HQVertexStreamManagerGL.cpp
	HQTextureManagerGL.cpp ../BaseImpl/HQTextureManagerBaseImpl.cpp 
	glew/src/glew.c
)


add_library(HQRendererGL  OBJECT
	${MAIN_SRC_FILES})
