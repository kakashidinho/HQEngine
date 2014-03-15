include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_OGG_INC_DIR}
    ${CMAKE_VORBIS_INC_DIR}
    ${CMAKE_OPENAL_INC_DIR}
)


set (MAIN_SRC_FILES 
	HQAudioBase.cpp openAL/HQAudioAL.cpp openAL/HQAudioStreamBufferAL.cpp HQAudioInternal.cpp
)


add_library(HQAudio OBJECT
	${MAIN_SRC_FILES}
)
