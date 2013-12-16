/*
 *  HQUtilMathCommon.h
 *
 *  Created by Kakashidinho on 5/28/11.
 *  Copyright 2011 LHQ. All rights reserved.
 *
 */

#ifndef HQ_UTIL_MATH_COMMON_H
#define HQ_UTIL_MATH_COMMON_H

#include "HQPlatformDef.h"

#if (defined HQ_STATIC_ENGINE) || defined IOS
#	define _STATICLINK
#endif

#if defined IOS || defined ANDROID || (defined HQ_WIN_PHONE_PLATFORM)
#	if	!defined CMATH && !defined NEON_MATH && !defined HQ_DIRECTX_MATH
#		if (defined HQ_WIN_PHONE_PLATFORM)
#			if 0
#				define HQ_DIRECTX_MATH
#			else
#				if defined (_M_ARM)
#					define NEON_MATH
#					define NEON_ASM
#				else
#					define HQ_DIRECTX_MATH 
#				endif
#			endif
#		else
#			if defined ANDROID
#				if defined __x86_64__ || defined __i386__ || defined __i686__
#					define SSE_MATH
#				elif defined __arm__
#					define NEON_MATH
#				else
#					define CMATH
#				endif
#			else
#				define NEON_MATH
#			endif
#		endif
#	endif
#endif

#if !defined CMATH && defined HQ_WIN_STORE_PLATFORM
#	if 1
#		define HQ_DIRECTX_MATH
#	else
#		if defined _M_ARM
#			define NEON_MATH
#			define NEON_ASM
#		endif
#	endif
#endif



#ifdef _STATICLINK
#	define HQ_UTIL_MATH_API
#else
#	if defined WIN32 || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#		ifdef Q_UTIL_EXPORTS
#			define HQ_UTIL_MATH_API __declspec(dllexport)
#		else
#			define HQ_UTIL_MATH_API __declspec(dllimport)
#		endif
#	else
#		define HQ_UTIL_MATH_API __attribute__ ((visibility("default")))
#	endif
#endif

//Utility macros
#ifndef SafeDelete
#define SafeDelete(a) {if(a != NULL) {delete a;a=NULL;}}
#endif
#ifndef SafeDeleteArray
#define SafeDeleteArray(a) {if(a != NULL) {delete[] a;a=NULL;}}
#endif

#ifndef HQ_FORCE_INLINE
#	ifdef _MSC_VER
#		if (_MSC_VER >= 1200)
#			define HQ_FORCE_INLINE __forceinline
#		else
#			define HQ_FORCE_INLINE __inline
#		endif
#	else
#		if defined DEBUG|| defined _DEBUG
#			define HQ_FORCE_INLINE inline
#		else
#			define HQ_FORCE_INLINE inline __attribute__ ((always_inline))
#		endif
#	endif
#endif

#ifdef CMATH
#define HQ_NO_NEED_ALIGN16

//#pragma message ("using plain C math")

#endif

#include "HQPrimitiveDataType.h"
#include "HQMiscDataType.h"
#include "HQMemAlignment.h"

#include <stdlib.h>
#include <math.h>

#if !defined (CMATH) && !defined (NEON_MATH) && !defined (HQ_DIRECTX_MATH)
#	define SSE_MATH
#endif


#ifdef SSE_MATH /*----SSE--------*/

//#pragma message ("using SSE math")

#include <xmmintrin.h> //Intel SSE intrinsics header
#include <emmintrin.h> //SSE2

#	ifdef SSE4_MATH
#include <smmintrin.h> //SSE4



#define SSE4_DP_MASK(b7 , b6 , b5 ,b4 , b3 ,b2 ,b1 , b0) ( (b7 << 7) | (b6 << 6) | (b5 << 5) | (b4 << 4) | (b3 << 3) | (b2 << 2) | (b1 << 1) | b0)

#	endif

typedef __m128 float4;
typedef __m128i int4;

#define hq_mm_copy_ps(src , imm) ( _mm_castsi128_ps (_mm_shuffle_epi32(_mm_castps_si128(src) , imm ) ) )

//shuffle macro for SSE shuffle instruction
#define SSE_SHUFFLE(x,y,z,w) (x<<6|y<<4|z<<2|w)
#define SSE_SHUFFLEL(w,z,y,x) (x<<6|y<<4|z<<2|w)

#elif defined NEON_MATH

//#	pragma message ("using Neon math")

#	if defined NEON_ASM
//#		pragma message ("using Neon assembly implementation")
#	endif


#include "HQNeonMatrixInline.h"
#include "HQNeonQuaternionInline.h"


#elif defined HQ_DIRECTX_MATH

//#	pragma message ("using DirectX math")

#include <DirectXMath.h>

#include "HQDXMatrixInline.h"
#include "HQDXQuaternionInline.h"

#endif

//#define INFCHECK 1

#if !defined HQ_EXPLICIT_ALIGN && !defined HQ_NO_NEED_ALIGN16
#	ifndef _MSC_VER//gcc stack alignment is buggy
#		define HQ_EXPLICIT_ALIGN
#	endif
#endif

//=======================================================
//sai số
#define EPSILON 0.00001f


//=======================================================
//đổi góc ra radian và độ
#define HQToRadian( degree ) ((degree) * (HQPiFamily::_PI / 180.0f))
#define HQToDegree( radian ) ((radian) * (180.0f / HQPiFamily::_PI))
//=======================================================

//hoán đổi 2 số
#define swapf(a,b) {hq_float32 t=a;a=b;b=t;}
#define swapui(a,b) {hq_uint32 t=a;a=b;b=t;}
#define swapi(a,b) {hq_int32 t=a;a=b;b=t;}
#define swapd(a,b) {hq_float64 t=a;a=b;b=t;}

//bình phương
#define sqr(a) (a*a)



//========================================================
//3D API type
typedef enum HQRenderAPI
{
	HQ_RA_D3D = 0,//direct3D
	HQ_RA_OGL = 1//openGL
} _HQRenderAPI;

//=======================================================
//số pi
namespace HQPiFamily
{
	static const hq_float32 _PI =  3.141592654f; //pi
	static const hq_float32 _1OVERPI = 0.318309886f; //1/pi
	static const hq_float32 _PIOVER2 = 1.570796327f; //pi/2
	static const hq_float32  _PIOVER3 = 1.047197551f; //pi/3
	static const hq_float32  _PIOVER4 =	 0.785398163f; //pi/4
	static const hq_float32  _PIOVER6 =	 0.523598775f; //pi/6
	static const hq_float32  _2PI   = 6.283185307f; //2*pi

	static const hq_float32 PI = _PI;
};



#endif
