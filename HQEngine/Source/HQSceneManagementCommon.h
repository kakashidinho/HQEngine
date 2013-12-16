#ifndef HQ_SCENE_MANAGEMENT_COMMON_H
#define HQ_SCENE_MANAGEMENT_COMMON_H

#include "HQPlatformDef.h"

#if defined HQ_STATIC_ENGINE || defined IOS
#	define _STATIC_LIB
#endif

#ifndef HQSCENEMANAGEMENT_API
#	ifdef _STATIC_LIB
#		define HQSCENEMANAGEMENT_API
#	else
#		if defined WIN32 || defined HQ_WIN_STORE_PLATFORM || defined HQ_WIN_PHONE_PLATFORM
#			ifdef HQSCENEMANAGEMENT_EXPORTS
#				define HQSCENEMANAGEMENT_API __declspec(dllexport)
#			else
#				define HQSCENEMANAGEMENT_API __declspec(dllimport)
#			endif
#		else
#				define HQSCENEMANAGEMENT_API __attribute__ ((visibility("default")))
#		endif
#	endif
#endif

#endif