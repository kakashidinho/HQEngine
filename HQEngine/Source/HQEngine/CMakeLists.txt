include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (TINY_XML_SRC_PATH ../../../ThirdParty-mod/tinyxml)

set (MAIN_SRC_FILES 
	HQEngineApp.cpp HQDefaultFileManager.cpp HQEngineCommonInternal.cpp 
	HQEngineEffectManagerImpl.cpp HQEngineResManagerImpl.cpp 
	HQEngineBaseWindow.cpp
	x11/HQEngineAppX11.cpp 
	x11/HQEngineWindowX11.cpp 
)

set (TINY_XML_SRC_FILES
	${TINY_XML_SRC_PATH}/tinystr.cpp 
	${TINY_XML_SRC_PATH}/tinyxml.cpp 
	${TINY_XML_SRC_PATH}/tinyxmlerror.cpp 
	${TINY_XML_SRC_PATH}/tinyxmlparser.cpp
)


add_library(HQEngineObjs OBJECT
	${MAIN_SRC_FILES}
	${TINY_XML_SRC_FILES}
)
