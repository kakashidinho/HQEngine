cmake_minimum_required (VERSION 2.8.8)
project (HQEngine)
# The version number.
set (Tutorial_VERSION_MAJOR 0)
set (Tutorial_VERSION_MINOR 1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)
find_package(X11_randr)

if (NOT X11_randr_FOUND)
	MESSAGE(FATAL_ERROR "X11 RandR extension's header and library not found!")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -msse -msse2 -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -msse -msse2 -fPIC")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG -D_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -D_DEBUG")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Source ${CMAKE_CURRENT_BINARY_DIR}/Source)

add_library (HQEngine SHARED 
	     $<TARGET_OBJECTS:HQEngineObjs> 
	     $<TARGET_OBJECTS:ImagesLoader> 
	     $<TARGET_OBJECTS:HQUtilMath> 
	     $<TARGET_OBJECTS:HQUtil> 
	     $<TARGET_OBJECTS:HQRendererGL> 
	     $<TARGET_OBJECTS:HQRendererFactory> 
	     $<TARGET_OBJECTS:HQSceneManagement>
	     $<TARGET_OBJECTS:HQAudio>)

	     
target_link_libraries(HQEngine 
	    ${OPENGL_gl_LIBRARY}
	    ${X11_X11_LIB}
	    ${X11_randr_LIBRARY}
	    ${CMAKE_VORBIS_LIB_DIR}/libvorbis.a
	    ${CMAKE_VORBIS_LIB_DIR}/libvorbisfile.a
	    ${CMAKE_OGG_LIB_DIR}/libogg.a
	    ${CMAKE_OPENAL_LIB_DIR}/libopenal.so
	    )
	    
add_executable(test ../Source/test/winmain.cpp ../Source/test/Game.cpp)
target_link_libraries(test HQEngine)

file(COPY ../Source/test/audio DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ../Source/test/meshes DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ../Source/test/shader DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ../Source/test/setting DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ../Source/test/script DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
