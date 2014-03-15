include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (MAIN_SRC_FILES 
	HQRenderer.cpp 
	HQRenderDeviceDebugLayer.cpp 
	HQReturnValDebugString.cpp 
)


add_library(HQRendererFactory OBJECT
	${MAIN_SRC_FILES})
