/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef BASEIMPLSHADERSTRING_HQFFEMUSHADERD3D1X_H_INCLUDED
#define BASEIMPLSHADERSTRING_HQFFEMUSHADERD3D1X_H_INCLUDED
const char HQFFEmuShaderD3D1x[] = "\
\r\n\
\r\n\
#define MAX_LIGHTS 4\r\n\
\r\n\
\r\n\
/* Vertex Attributes */\r\n\
\r\n\
//uniform vertex input structure\r\n\
struct Vinput\r\n\
{\r\n\
	float3 position	: VPOSITION;\r\n\
	float4 color	: VCOLOR;\r\n\
	float3 normal	: VNORMAL;\r\n\
	float2 texcoords: VTEXCOORD0;\r\n\
};\r\n\
\r\n\
cbuffer Parameters : register (b0)\r\n\
{\r\n\
	/* Matrix Uniforms */\r\n\
\r\n\
	float4x4 uMvpMatrix;\r\n\
	float4x4 uWorldMatrix;\r\n\
\r\n\
	/* Light Uniforms */\r\n\
	float4  uLightPosition    [MAX_LIGHTS];\r\n\
	float4  uLightAmbient     [MAX_LIGHTS];\r\n\
	float4  uLightDiffuse     [MAX_LIGHTS];\r\n\
\r\n\
	float4  uLightSpecular    [MAX_LIGHTS];\r\n\
\r\n\
	float3  uLightAttenuation [MAX_LIGHTS];// C struct need to add padding float at the end of each element\r\n\
\r\n\
	/* Global ambient color */\r\n\
	float4  uAmbientColor;\r\n\
\r\n\
	/* Material Uniforms */\r\n\
	float4  uMaterialAmbient;\r\n\
	float4  uMaterialEmission;\r\n\
	float4  uMaterialDiffuse;\r\n\
	float4  uMaterialSpecular;\r\n\
	float uMaterialShininess;\r\n\
	\r\n\
	/* Eye position */\r\n\
	float3 uEyePos;\r\n\
	\r\n\
	/* For light 0 - 3 */\r\n\
	int4  uUseLight;\r\n\
	\r\n\
	/* Normalize normal? */\r\n\
	int uNormalize;\r\n\
	\r\n\
\r\n\
\r\n\
};//cbuffer Parameters\r\n\
\r\n\
//uniform vertex output structure\r\n\
struct Voutput\r\n\
{\r\n\
	float4 position	: SV_Position;\r\n\
	float4 color	: COLOR;\r\n\
	float2 texcoords: TEXCOORD0;\r\n\
};\r\n\
\r\n\
float4 lightEquation(int lidx, in float3 gWorldNormal, in float3 gWorldPos)\r\n\
{		\r\n\
	float4 color = float4(0.0, 0.0, 0.0, 0.0);\r\n\
	\r\n\
	float att = 1.0;\r\n\
	float3 lightDir;\r\n\
	\r\n\
	\r\n\
	if( uLightPosition[lidx].w == 0.0) // Directional light\r\n\
	{\r\n\
		lightDir = -uLightPosition[lidx].xyz;\r\n\
	}\r\n\
	else\r\n\
	{\r\n\
		lightDir = uLightPosition[lidx].xyz - gWorldPos.xyz;\r\n\
		float dir = length(lightDir);\r\n\
		att = 1.0 / (uLightAttenuation[lidx].x + uLightAttenuation[lidx].y * dir + uLightAttenuation[lidx].z * dir * dir);\r\n\
		lightDir = normalize(lightDir);\r\n\
	}\r\n\
	\r\n\
	if(att >= 0.0 )\r\n\
	{\r\n\
		color += uLightAmbient[lidx] * uMaterialAmbient;\r\n\
		\r\n\
		//Compute cos(Light, Normal)\r\n\
		float NdotL = max(dot(normalize(gWorldNormal), lightDir), 0.0);\r\n\
		color += NdotL * uLightDiffuse[lidx] * uMaterialDiffuse;\r\n\
		\r\n\
		//Compute cos(hvec, Normal)\r\n\
#ifdef USE_SPECULAR\r\n\
		{\r\n\
			float3 hvec = normalize(lightDir + normalize(uEyePos - gWorldPos));\r\n\
			float NdotH = dot(gWorldNormal, hvec);\r\n\
			if(NdotH > 0.0)\r\n\
			{\r\n\
				color += pow(NdotH, uMaterialShininess) * uLightSpecular[lidx] * uMaterialSpecular;\r\n\
			}\r\n\
		}\r\n\
#endif//#ifdef USE_SPECULAR\r\n\
		color *= att;\r\n\
	}\r\n\
	return color;\r\n\
}\r\n\
\r\n\
float4 computeLighting( in float3 gWorldNormal, in float3 gWorldPos)\r\n\
{\r\n\
	float4 color = uMaterialEmission + uMaterialAmbient * uAmbientColor;\r\n\
\r\n\
	for ( int i = 0 ; i < MAX_LIGHTS; ++i)\r\n\
	{\r\n\
		if ( uUseLight[i] )\r\n\
		{\r\n\
			color += lightEquation(i, gWorldNormal, gWorldPos) ;\r\n\
		}\r\n\
	}\r\n\
	color.a = uMaterialDiffuse.a;\r\n\
	return color;\r\n\
}\r\n\
\r\n\
/*---------------vertex shader----------*/\r\n\
Voutput VS(in Vinput vin)\r\n\
{\r\n\
	Voutput vout;\r\n\
	\r\n\
	float3 gWorldNormal;\r\n\
	float3 gWorldPos;\r\n\
	float4 gColor;\r\n\
\r\n\
	float4 vposition = float4(vin.position, 1.0);\r\n\
\r\n\
	vout.position = mul( uMvpMatrix, vposition  ) ;\r\n\
	\r\n\
	gWorldPos = mul( uWorldMatrix, vposition  ).xyz;\r\n\
	\r\n\
	gColor = vin.color;\r\n\
	\r\n\
#ifdef USE_LIGHTING\r\n\
	{\r\n\
		gWorldNormal = mul( uWorldMatrix , float4(vin.normal, 0.0)).xyz;\r\n\
		if(uNormalize)\r\n\
			gWorldNormal = normalize(gWorldNormal);\r\n\
	\r\n\
		vout.color = computeLighting(gWorldNormal, gWorldPos) * gColor;\r\n\
	}\r\n\
#else\r\n\
		vout.color = gColor;\r\n\
#endif //#ifdef USE_LIGHTING\r\n\
	\r\n\
	\r\n\
#ifdef USE_TEXTURE\r\n\
	{\r\n\
		vout.texcoords = vin.texcoords;\r\n\
	}\r\n\
#endif\r\n\
	\r\n\
	vout.color.rgb  = clamp(vout.color.rgb, 0.0, 1.0);\r\n\
	vout.color.a = gColor.a;\r\n\
	\r\n\
	return vout;\r\n\
}\r\n\
\r\n\
\r\n\
/*---------------pixel shader----------*/\r\n\
\r\n\
Texture2D texture0 : register (t0);\r\n\
SamplerState sampler0 : register (s0);\r\n\
\r\n\
float4 PS (in Voutput pin) :SV_Target\r\n\
{\r\n\
	float4 color = pin.color;\r\n\
#ifdef USE_TEXTURE\r\n\
	{\r\n\
		float4 texColor = (texture0.Sample ( sampler0 , pin.texcoords));\r\n\
		color.xyz *=  texColor.xyz;\r\n\
		color.a = texColor.a;\r\n\
	}\r\n\
#endif\r\n\
		\r\n\
	return color;\r\n\
}\n";
#endif
