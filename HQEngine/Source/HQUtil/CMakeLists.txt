include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (MAIN_SRC_FILES 
	pthread/HQMutexPthread.cpp pthread/HQSemaphorePthread.cpp 
	pthread/HQThreadPthread.cpp linux/HQTimerLinux.cpp 
	pthread/HQConditionVariablePThread.cpp
	HQLogStream.cpp
)


add_library(HQUtil  OBJECT
	${MAIN_SRC_FILES})
