// stdafx.cpp : source file that includes just the standard includes
// HQUtilMath.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "HQUtilMathPCH.h"

// TODO: reference any additional headers you need in STDAFX.H
// and not in this file

#ifdef SSE_MATH
#	pragma message("using SSE Math")
#elif defined NEON_MATH
#	pragma message("using Neon Math")
#	if defined NEON_ASM
#		pragma message("using Neon ASM")
#	endif
#elif defined HQ_DIRECTX_MATH
#	pragma message("using DirectX Math")
#elif defined CMATH
#	pragma message("using C Math")
#endif
