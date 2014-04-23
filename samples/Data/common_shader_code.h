#ifndef COMMON_SHADER_CODE_H
#define COMMON_SHADER_CODE_H

#define PI 3.141592654f

#ifdef HQEXT_CG
#pragma pack_matrix(column_major)

struct vPNIn{
	float3 iPos : VPOSITION;
	float3 iNormal : VNORMAL;
};


float3 worldTransformCoords(float3 posL, float4x3 worldMatrix){
	return mul(float4(posL, 1.0), worldMatrix);
}

float3 worldTransformNormal(float3 normalL, float4x3 worldMatrix){
	float3 normalW = mul(float4(normalL, 0.0), worldMatrix);
	return normalize(normalW);
}

float4 clipTransformCoords(float3 posW, float4x4 viewMatrix, float4x4 projMatrix)
{
	float4 posV = mul(float4(posW, 1.0), viewMatrix);
	return mul(posV, projMatrix);
}

void transformWVP(float3 posL, 
	float4x3 worldMatrix, 
	float4x4 viewMatrix, 
	float4x4 projMatrix,
	out float3 posW,
	out float4 posH)
{

	posW = worldTransformCoords(posL, worldMatrix);
	posH = clipTransformCoords(posW, viewMatrix, projMatrix);
}

#else
//TO DO
#endif//#ifdef HQEXT_CG


#endif