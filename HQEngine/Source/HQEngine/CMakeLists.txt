include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (TINY_XML_SRC_PATH ../../../ThirdParty-mod/tinyxml)


add_custom_target (
	parsers_gen_code
	COMMAND chmod +x ${CMAKE_CURRENT_SOURCE_DIR}/generate_parsers.sh
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_parsers.sh
)


set (MAIN_SRC_FILES 
	HQEngineApp.cpp HQDefaultFileManager.cpp HQEngineCommonInternal.cpp 
	HQEngineEffectManagerImpl.cpp HQEngineResManagerImpl.cpp 
	HQEngineBaseWindow.cpp
	x11/HQEngineAppX11.cpp 
	x11/HQEngineWindowX11.cpp 
	HQEngineResParserCommon.cpp
	gen_res_script_tokens.cpp
	gen_res_script_parser.cpp
	gen_effect_script_tokens.cpp
	gen_effect_script_parser.cpp
)

add_library(HQEngineObjs OBJECT
	${MAIN_SRC_FILES}
)

add_dependencies(HQEngineObjs parsers_gen_code)
