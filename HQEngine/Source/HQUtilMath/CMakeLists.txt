include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (MAIN_SRC_FILES 
	HQUtilMathPCH.cpp HQAABB.cpp HQBSP.cpp HQMatrix3x4.cpp HQMatrix4.cpp 
	HQOBB.cpp HQPlane.cpp HQPolygon.cpp HQPrimeNumber.cpp 		
	HQQuaternion.cpp HQRay.cpp HQSphere.cpp HQVector.cpp 
)


add_library(HQUtilMath OBJECT
	${MAIN_SRC_FILES})
