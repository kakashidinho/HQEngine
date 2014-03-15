include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (MAIN_SRC_FILES 
	helperFunctions.cpp HQMeshNode.cpp HQMeshNode_Animation.cpp HQSceneNode.cpp
	HQCamera.cpp
)


add_library(HQSceneManagement  OBJECT
	${MAIN_SRC_FILES})
