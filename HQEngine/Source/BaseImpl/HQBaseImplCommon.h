#ifndef _HQ_COMMON_H_
#define _HQ_COMMON_H_

#include "../HQPlatformDef.h"

/*---------------flags--------------------*/
#ifdef ANDROID
#	define DEVICE_LOST_POSSIBLE
#endif

#define RUNNING 0x1
#define RENDER_BEGUN 0x2
#define WINDOWED 0x4
#define DEVICE_LOST 0x8
#define VSYNC_ENABLE 0x10
#define USEVSHADER 0x20
#define USEGSHADER 0x40
#define USEPSHADER 0x80
#define USEFSHADER 0x80
#define USESHADER (USEVSHADER | USEGSHADER | USEPSHADER)

#define CLEAR_COLOR_OR_DEPTH_CHANGED 0x100
#define INACTIVE 0x200





#if !defined WIN32
#	ifndef DWORD
#		define DWORD hq_uint32
#	endif
#	ifndef WORD
#		define WORD hq_ushort16
#	endif
#	ifndef UINT
#		define UINT hq_uint32
#	endif
#endif



#ifndef SafeDelete
#	define SafeDelete(p){if(p != NULL){delete p;p=NULL;}}
#endif
#ifndef SafeDeleteArray
#	define SafeDeleteArray(p){if(p != NULL){delete[] p;p=NULL;}}
#endif

#ifndef SafeDeleteTypeCast
#	define SafeDeleteTypeCast(casting_type, ptr) {if(ptr != NULL) {delete static_cast<casting_type> (ptr); ptr=NULL; } }
#endif

#ifndef SafeRelease
#define SafeRelease(p) {if(p){p->Release();p=0;}}
#endif
#ifndef SafeDelete
#define SafeDelete(p){if(p){delete p;p=0;}}
#endif
#ifndef SafeDeleteArray
#define SafeDeleteArray(p){if(p){delete[] p;p=0;}}
#endif

#define COLOR_ARGB(a,r,g,b) (((a&0xff)<<24) | ((r&0xff)<<16) | ((g&0xff)<<8) | ((b&0xff)) )
#define COLOR_XRGB(r,g,b) COLOR_ARGB(1,r,g,b)
#define COLOR_COLORVALUE(a,r,g,b) ((((DWORD)(a*255)&0xff)<<24) |\
								  (((DWORD)(r*255)&0xff)<<16) |\
								  (((DWORD)(g*255)&0xff)<<8) |\
								  (((DWORD)(b*255)&0xff)) )

#define GetRfromARGB(c) ((c & 0x00ff0000)>>16)
#define GetGfromARGB(c) ((c & 0x0000ff00)>>8)
#define GetBfromARGB(c) ((c & 0x000000ff))
#define GetAfromARGB(c) ((c & 0xff000000)>>24)


/*-------------------------------*/

#endif