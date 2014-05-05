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
	float3 oNormalW: TEXCOORD2;//world space normal
};


//vertex shader
vOut VS(in vPNIn input){
	vOut output;
	
	float3 posW;
	float4 posH;
	float3 normalW; 
	
	transformWVP(input.iPos, worldMat, lightViewMat, lightProjMat, posW, posH);
	normalW = worldTransformNormal(input.iNormal, worldMat );
	
	//projected position
	output.oPos = posH;
	
	//world space position
	output.oPosW = posW;
	
	//world space normal
	output.oNormalW = normalW;
	
	return output;
};


cbuffer materials : register (b2) {
	float4 materialDiffuse[7];
} ;

cbuffer materialIndex : register (b3) {
	int materialID;
};

cbuffer lightProperties : register (b4) {
	
	float3 lightPosition;
	float3 lightDirection;
	float4 lightDiffuse;
	float3 lightFalloff_cosHalfAngle_cosHalfTheta;
};


struct pOut{
	float2 depth_materialID: SV_Target00;//distance to camera and material ID
	float4 posW: SV_Target01;//world space position
	float4 normalW: SV_Target02;//world space normal
	float4 flux: SV_Target03;//flux
};

//pixel shader
pOut PS(
	in float4 vPos : SV_Position,
	in float3 posW : TEXCOORD0,//world space position
	in float3 normalW: TEXCOORD2 //world space normal
	)
{
	pOut output;
	
	//re-normalize normal
	normalW = normalize(normalW);

	output.depth_materialID.x = length(posW - lightPosition.xyz);//distance to light
	output.depth_materialID.y = materialID;//material ID
	
	output.posW = float4(posW, 1.0);//world space position 
	
	output.normalW = float4(normalW, 1.0);//world space normal

	//calculate flux
	float3 lightVec = normalize(posW - lightPosition);
	float spot = calculateSpotLightFactor(lightVec, lightDirection, lightFalloff_cosHalfAngle_cosHalfTheta);
	
	output.flux = spot * lightDiffuse * materialDiffuse[materialID];
	
	return output;
}