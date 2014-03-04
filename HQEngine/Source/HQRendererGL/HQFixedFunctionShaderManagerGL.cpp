/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"
#include "glHeaders.h"
#include "HQFixedFunctionShaderManagerGL.h"
#include "HQDeviceGL.h"

#define GetRfromBGRA(c) ((c & (0xff << 16)) >> 16)
#define GetGfromBGRA(c) ((c & (0xff << 8)) >> 8)
#define GetBfromBGRA(c) (c & 0xff)
#define GetAfromBGRA(c) ((c & (0xff << 24)) >> 24)

HQFixedFunctionShaderManagerGL::HQFixedFunctionShaderManagerGL(HQLogStream *logStream, bool flushLog)
:   HQLoggableObject(logStream , "Fixed Function Shader Manager:" ,flushLog),
	world(HQMatrix4::New()) , view(HQMatrix4::New())
{
	materialSpecular.r = materialSpecular.g = materialSpecular.b = 0;
	materialSpecular.a = 1.0f;
	/*----default states---------*/
	glMaterialfv(GL_FRONT_AND_BACK , GL_SPECULAR , materialSpecular);//black
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT , materialSpecular);//black
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_LIGHTING);
	glDisable(GL_NORMALIZE);

	Log("Init done!");
}

HQFixedFunctionShaderManagerGL::~HQFixedFunctionShaderManagerGL()
{
	SafeDelete(world);
	SafeDelete(view);
	Log("Released!");
}

HQReturnVal HQFixedFunctionShaderManagerGL::SetUniformInt(hq_uint32 stateType,
								 const hq_int32* pValue,
								 hq_uint32 numElements)
{
	switch(stateType)
	{
	case HQ_LIGHT0: case HQ_LIGHT1: case HQ_LIGHT2:
	case HQ_LIGHT3: 
		this->SetLight(stateType , (HQFFLight *) pValue);
		break;
	case HQ_LIGHT0_ENABLE : case HQ_LIGHT1_ENABLE: case HQ_LIGHT2_ENABLE: 
	case HQ_LIGHT3_ENABLE: 
		if (*pValue == HQ_TRUE)
			glEnable(GL_LIGHT0 - HQ_LIGHT0_ENABLE + stateType);
		else 
			glDisable(GL_LIGHT0 - HQ_LIGHT0_ENABLE + stateType);
		break;
	case HQ_MATERIAL:
		this->SetMaterial((HQFFMaterial *)pValue);
		break;
	case HQ_TEXTURE_ENABLE:
	 	static_cast<HQTextureManagerGL*> (g_pOGLDev->GetTextureManager())->ActiveTextureUnit(0);
		if (*pValue == HQ_TRUE)
		{
			glEnable(GL_TEXTURE_2D);
		}
		else
		{
			glDisable(GL_TEXTURE_2D);
		}
		break;
	case HQ_AMBIENT:
		{
			HQColorui color = *((HQColorui*)pValue);
			glLightModelfv(GL_LIGHT_MODEL_AMBIENT , 
				HQColorRGBAi(GetRfromBGRA(color) , GetGfromBGRA(color) , GetBfromBGRA(color), GetAfromBGRA(color) ));
		}
		break;
	case HQ_LIGHTING_ENABLE:
		if (*pValue == HQ_TRUE)
			glEnable(GL_LIGHTING);
		else
			glDisable(GL_LIGHTING);
		break;
	case HQ_SPECULAR_ENABLE:
		if (*pValue == HQ_TRUE)
			glMaterialfv(GL_FRONT_AND_BACK , GL_SPECULAR , this->materialSpecular);
		else
		{
			GLfloat black[] = {0,0,0,1};
			glMaterialfv(GL_FRONT_AND_BACK , GL_SPECULAR , black);
		}
		break;
	case HQ_NORMALIZE_NORMALS:
		if (*pValue == HQ_TRUE)
			glEnable(GL_NORMALIZE);
		else
			glDisable(GL_NORMALIZE);
		break;
	default:
		return HQ_FAILED;
	}
	return HQ_OK;
}

HQReturnVal HQFixedFunctionShaderManagerGL::SetUniformMatrix(hq_uint32 parameterIndex,
								   const HQBaseMatrix4* pMatrice,
								   hq_uint32 numMatrices)
{
	switch(parameterIndex)
	{
	case HQ_WORLD:
		glMatrixMode(GL_MODELVIEW);
		memcpy(this->world, pMatrice, sizeof(HQMatrix4));
		this->SetModelViewMatrix();
		break;
	case HQ_VIEW:
		memcpy(this->view, pMatrice, sizeof(HQMatrix4));
		//reset light position
		for (unsigned int i = 0 ; i < 8 ; ++i)
			this->SetLightPosition(i);//auto set glMatrixMode(GL_MODELVIEW);
		this->SetModelViewMatrix();
		break;
	case HQ_PROJECTION:
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(pMatrice->m);
		break;
	default:
		return HQ_FAILED;
	}
	return HQ_OK;
}

void HQFixedFunctionShaderManagerGL::SetModelViewMatrix()
{
	HQ_DECL_STACK_MATRIX4(model_view);
	HQMatrix4Multiply(this->world, this->view , &model_view);
	glLoadMatrixf(model_view);

}

void HQFixedFunctionShaderManagerGL::SetLight(unsigned int light , HQFFLight* lightInfo)
{
	GLenum gllight = GL_LIGHT0 + light;
	glLightfv(gllight , GL_AMBIENT , lightInfo->ambient);
	glLightfv(gllight , GL_DIFFUSE , lightInfo->diffuse);
	glLightfv(gllight , GL_SPECULAR , lightInfo->specular);
	glLightf(gllight , GL_CONSTANT_ATTENUATION , lightInfo->attenuation0);
	glLightf(gllight , GL_LINEAR_ATTENUATION , lightInfo->attenuation1);
	glLightf(gllight , GL_QUADRATIC_ATTENUATION , lightInfo->attenuation2);
	/*---light direction---------*/
	
	if (lightInfo->type == HQ_LIGHT_POINT)
	{
		memcpy(&this->lightPosition[light], &lightInfo->position , sizeof(HQFloat3));
		this->lightPosition[light][3] = 1.0f;
	}
	else//directional
	{
		this->lightPosition[light][0] = -lightInfo->direction.x;
		this->lightPosition[light][1] = -lightInfo->direction.y;
		this->lightPosition[light][2] = -lightInfo->direction.z;
		this->lightPosition[light][3] = 0.0f;
	}
	
	this->SetLightPosition(light);
}
void HQFixedFunctionShaderManagerGL::SetMaterial(HQFFMaterial *material)
{
	this->materialSpecular = material->specular;//cache specular material

	glMaterialfv(GL_FRONT_AND_BACK , GL_AMBIENT , material->ambient);
	glMaterialfv(GL_FRONT_AND_BACK , GL_DIFFUSE , material->diffuse);
	glMaterialfv(GL_FRONT_AND_BACK , GL_SPECULAR , material->specular);
	glMaterialfv(GL_FRONT_AND_BACK , GL_EMISSION , material->emissive);
	glMaterialf(GL_FRONT_AND_BACK , GL_SHININESS , material->power);
}

void HQFixedFunctionShaderManagerGL::SetLightPosition(unsigned int light)
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadMatrixf(*this->view);
	glLightfv(light + GL_LIGHT0, GL_POSITION , this->lightPosition[light]);
	glPopMatrix();
}
