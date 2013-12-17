/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef BASEIMPLSHADERSTRING_HQFFEMUSHADERGL_H_INCLUDED
#define BASEIMPLSHADERSTRING_HQFFEMUSHADERGL_H_INCLUDED
const char HQFFEmuShaderGL[] = "\
\n\
\n\
#define MAX_LIGHTS 4\n\
\n\
#ifdef VERTEX_SHADER\n\
/* Vertex Attributes */\n\
\n\
//uniform vertex input structure\n\
attribute vec3 aPosition;\n\
attribute vec4 aColor;\n\
attribute vec3 aNormal;\n\
attribute vec2 aTexcoords;\n\
\n\
/* Matrix Uniforms */\n\
\n\
uniform mat4 uMvpMatrix;\n\
uniform mat4 uWorldMatrix;\n\
\n\
/* Light Uniforms */\n\
uniform vec4  uLightPosition    [MAX_LIGHTS];\n\
uniform vec4  uLightAmbient     [MAX_LIGHTS];\n\
uniform vec4  uLightDiffuse     [MAX_LIGHTS];\n\
\n\
uniform vec4  uLightSpecular    [MAX_LIGHTS];\n\
\n\
uniform vec4  uLightAttenuation [MAX_LIGHTS];// w is ignored. C struct need to add padding float at the end of each element\n\
\n\
/* Global ambient color */\n\
uniform vec4  uAmbientColor;\n\
\n\
/* Material Uniforms */\n\
uniform vec4  uMaterialAmbient;\n\
uniform vec4  uMaterialEmission;\n\
uniform vec4  uMaterialDiffuse;\n\
uniform vec4  uMaterialSpecular;\n\
uniform float uMaterialShininess;\n\
\n\
/* Eye position */\n\
uniform vec3 uEyePos;\n\
\n\
/* For light 0 - 3 */\n\
uniform vec4  uUseLight;\n\
\n\
/* Normalize normal? */\n\
uniform float uNormalize;\n\
	\n\
\n\
//uniform vertex output structure\n\
varying lowp vec4 vColor;\n\
varying mediump	vec2 vTexcoords;\n\
\n\
vec4 lightEquation(int lidx, vec3 gWorldNormal, vec3 gWorldPos)\n\
{		\n\
	vec4 color = vec4(0.0, 0.0, 0.0, 0.0);\n\
	\n\
	float att = 1.0;\n\
	vec3 lightDir;\n\
	\n\
	\n\
	if( uLightPosition[lidx].w == 0.0) // Directional light\n\
	{\n\
		lightDir = -uLightPosition[lidx].xyz;\n\
	}\n\
	else\n\
	{\n\
		lightDir = uLightPosition[lidx].xyz - gWorldPos.xyz;\n\
		float dir = length(lightDir);\n\
		att = 1.0 / (uLightAttenuation[lidx].x + uLightAttenuation[lidx].y * dir + uLightAttenuation[lidx].z * dir * dir);\n\
		lightDir = normalize(lightDir);\n\
	}\n\
	\n\
	if(att >= 0.0 )\n\
	{\n\
		color += uLightAmbient[lidx] * uMaterialAmbient;\n\
		\n\
		//Compute cos(Light, Normal)\n\
		float NdotL = max(dot(normalize(gWorldNormal), lightDir), 0.0);\n\
		color += NdotL * uLightDiffuse[lidx] * uMaterialDiffuse;\n\
		\n\
		//Compute cos(hvec, Normal)\n\
#ifdef USE_SPECULAR\n\
		{\n\
			vec3 hvec = normalize(lightDir + normalize(uEyePos - gWorldPos));\n\
			float NdotH = dot(gWorldNormal, hvec);\n\
			if(NdotH > 0.0)\n\
			{\n\
				color += pow(NdotH, uMaterialShininess) * uLightSpecular[lidx] * uMaterialSpecular;\n\
			}\n\
		}\n\
#endif//#ifdef USE_SPECULAR\n\
		color *= att;\n\
	}\n\
	return color;\n\
}\n\
\n\
vec4 computeLighting( vec3 gWorldNormal, vec3 gWorldPos)\n\
{\n\
	vec4 color = uMaterialEmission + uMaterialAmbient * uAmbientColor;\n\
\n\
	for ( int i = 0 ; i < MAX_LIGHTS; ++i)\n\
	{\n\
		if ( uUseLight[i] == 1.0)\n\
		{\n\
			color += lightEquation(i, gWorldNormal, gWorldPos) ;\n\
		}\n\
	}\n\
	color.w = uMaterialDiffuse.w;\n\
	return color;\n\
}\n\
\n\
/*---------------vertex shader----------*/\n\
void main()\n\
{	\n\
	vec3 gWorldNormal;\n\
	vec3 gWorldPos;\n\
	vec4 gColor;\n\
\n\
	vec4 vposition = vec4(aPosition, 1.0);\n\
\n\
	gl_Position = vposition *  uMvpMatrix;\n\
	\n\
	gWorldPos = ( vposition * uWorldMatrix).xyz;\n\
	\n\
	gColor = aColor;\n\
	\n\
#ifdef USE_LIGHTING\n\
	{\n\
		gWorldNormal = ( vec4(aNormal, 0.0) * uWorldMatrix  ).xyz;\n\
		if(uNormalize == 1.0)\n\
			gWorldNormal = normalize(gWorldNormal);\n\
	\n\
		vColor = computeLighting(gWorldNormal, gWorldPos) * gColor;\n\
	}\n\
#else\n\
		vColor = gColor;\n\
#endif //#ifdef USE_LIGHTING\n\
	\n\
	\n\
#ifdef USE_TEXTURE\n\
	{\n\
		vTexcoords = aTexcoords;\n\
	}\n\
#endif\n\
	\n\
	vColor.xyz  = clamp(vColor.xyz, 0.0, 1.0);\n\
	vColor.w = gColor.w;\n\
}\n\
\n\
#elif defined FRAGMENT_SHADER\n\
\n\
\n\
/*---------------pixel shader----------*/\n\
varying lowp vec4 vColor;\n\
varying mediump	vec2 vTexcoords;\n\
\n\
uniform sampler2D texture0 ;\n\
\n\
void main ()\n\
{\n\
	gl_FragColor = vColor;\n\
#ifdef USE_TEXTURE\n\
	\n\
		lowp vec4 texColor = texture2D ( texture0 , vTexcoords);\n\
		gl_FragColor.xyz *=  texColor.xyz;\n\
		gl_FragColor.w = texColor.w;\n\
	\n\
#endif\n\
}\n\
\n\
#endif//#ifdef VERTEX_SHADER\n";
#endif
