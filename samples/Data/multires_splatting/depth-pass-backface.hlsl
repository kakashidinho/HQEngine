/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../common_shader_code.h"

#define NORMALIZE_RESULTS 0


cbuffer transform : register (b0) {
	float4x3 worldMat;
};

cbuffer lightView : register (b1) {
	float4x4 lightViewMat;//light camera's view matrix
	float4x4 lightProjMat;//light camera's projection matrix
};

#define HALF_PIXEL_OFFSET 0.5 / 512

struct vOut{
	float4 oPos : SV_Position;
	float3 oPosW : TEXCOORD0;//world space position
};

//vertex shader
vOut VS(in vPNIn input){
	vOut output;
	
	float3 posW;
	float4 posH;
	
	transformWVP(input.iPos, worldMat, lightViewMat, lightProjMat, posW, posH);
	
	//projected position
	output.oPos = posH;
	
	//world space position
	output.oPosW = posW;
	
	return output;
}

cbuffer lightProperties : register (b4) {
	float3 lightPosition;
};

struct pOut{
	float depth: SV_Target00;//distance to light
};

//pixel shader
pOut PS(
	in float4 vPos : SV_Position,
	in float3 posW : TEXCOORD0//world space position
	)
{
	pOut output;
	
	output.depth = length(posW - lightPosition.xyz);//distance to light
	
	return output;
}