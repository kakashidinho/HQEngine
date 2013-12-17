/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _D3D1x_CLEAR_VP_SHADER_CODE_
#define _D3D1x_CLEAR_VP_SHADER_CODE_

#define HQ_D3D_CLEAR_VP_USE_GS 0
#define HQ_D3D_CLEAR_VP_USE_BYTE_CODE 1

const char g_clearShaderCode[]= 
#if HQ_D3D_CLEAR_VP_USE_GS
"\
struct VS_OUT\n\
{\n\
	float4 color : COLOR;\n\
	float depth : DEPTH;\n\
};\n\
\n\
struct GS_OUT\n\
{\n\
	float4 color : COLOR;\n\
	float4 position : SV_POSITION;\n\
};\n\
\n\
VS_OUT VS(in float4 color : COLOR,\n\
				   in float depth : DEPTH)\n\
{\n\
	VS_OUT vout;\n\
	vout.color = color;\n\
	vout.depth = depth;\n\
	return vout;\n\
}\n\
\n\
[maxvertexcount(4)]\n\
void GS(point VS_OUT gIn[1],\n\
		inout TriangleStream<GS_OUT> triStream)\n\
{\n\
	GS_OUT points[4];\n\
\n\
	points[0].position = float4(-1.0f , -1.0f , gIn[0].depth , 1.0f);\n\
\n\
	points[1].position = float4(-1.0f , 1.0f , gIn[0].depth , 1.0f);\n\
\n\
	points[2].position = float4(1.0f , -1.0f , gIn[0].depth , 1.0f);\n\
\n\
	points[3].position = float4(1.0f , 1.0f , gIn[0].depth , 1.0f);\n\
\n\
\n\
	[unroll]\n\
	for (int i = 0 ; i < 4 ; ++i)\n\
	{\n\
		points[i].color = gIn[0].color;\n\
		triStream.Append(points[i]);\n\
	}\n\
}\n\
\n\
float4 PS(in GS_OUT pIn) :SV_Target\n\
{\n\
	return pIn.color;\n\
}\n\
"
#else//#if !HQ_D3D_CLEAR_VP_USE_GS

#if !HQ_D3D_CLEAR_VP_USE_BYTE_CODE
"cbuffer Parameters : register(b0)\
{\
	float4 COLOR;\
	float DEPTH;\
};\
\
struct VS_OUT\
{\
	float4 position : SV_Position;\
	float4 color : COLOR;\
};\
\
VS_OUT VS(in float2 position : POSITION)\
{\
	VS_OUT vout;\
	vout.position = float4(position.x, position.y, DEPTH, 1.0f);\
	vout.color = COLOR;\
	return vout;\
}\
\
float4 PS(in VS_OUT pIn) :SV_Target\
{\
	return pIn.color;\
}\
"
#else
""//empty string, we will use byte code instead
#endif//#if !HQ_D3D_CLEAR_VP_USE_BYUTE_CODE

#endif//#if HQ_D3D_CLEAR_VP_USE_GS
;

#if HQ_D3D_CLEAR_VP_USE_BYTE_CODE
#include "compiledBaseImplShader\HQClearViewportShaderCodeD3D1x_VS.h"
#include "compiledBaseImplShader\HQClearViewportShaderCodeD3D1x_PS.h"
#endif

#endif
