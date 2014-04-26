/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D11PCH.h"
#include "HQShaderD3D11.h"

#if !(defined HQ_WIN_STORE_PLATFORM || defined HQ_WIN_PHONE_PLATFORM)
#define USE_SHADER_BYTE_CODE 1
#else
#define USE_SHADER_BYTE_CODE 1
#endif

#if !USE_SHADER_BYTE_CODE

#error need implement

#include "../BaseImpl/BaseImplShaderString/HQFFEmuShaderD3D1x.h"

#else

#if (defined _DEBUG || defined DEBUG) && defined HQ_WIN_STORE_PLATFORM
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_light_spec_tex_debug.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_light_spec_notex_debug.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_light_nospec_tex_debug.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_light_nospec_notex_debug.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_nolight_nospec_tex_debug.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_nolight_nospec_notex_debug.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_PS_tex_debug.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_PS_notex_debug.h"
#else
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_light_spec_tex.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_light_spec_notex.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_light_nospec_tex.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_light_nospec_notex.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_nolight_nospec_tex.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_VS_nolight_nospec_notex.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_PS_tex.h"
#include "../BaseImpl/compiledBaseImplShader/HQFFEmuShaderD3D1x_PS_notex.h"
#endif

#endif

#define MAX_LIGHTS 4

#define PARAMETERS_DIRTY 0x1
#define PROGRAM_DIRTY 0x2
#define FF_ACTIVE		  0x4
#define PROGRAM_SWITCHING 0x8

struct HQFixedFunctionParamenters: public HQA16ByteObject
{
	/* Matrix Uniforms */

	HQBaseMatrix4 uMvpMatrix;
	HQBaseMatrix4 uWorldMatrix;

	/* Light Uniforms */
	HQFloat4  uLightPosition    [MAX_LIGHTS];
	HQFloat4  uLightAmbient     [MAX_LIGHTS];
	HQFloat4  uLightDiffuse     [MAX_LIGHTS];
	HQFloat4  uLightSpecular    [MAX_LIGHTS];
	HQFloat4  uLightAttenuation [MAX_LIGHTS];// w is ignored. C struct need to add padding float at the end of each element
	
	/* Global ambient color */
	HQFloat4  uAmbientColor;

	/* Material Uniforms */
	HQFloat4  uMaterialAmbient;
	HQFloat4  uMaterialEmission;
	HQFloat4  uMaterialDiffuse;
	HQFloat4  uMaterialSpecular;
	float uMaterialShininess;
	
	/* Eye position */
	HQFloat3 uEyePos;

	/* For light 0 - 3 */
	int  uUseLight[4];

	/* Normalize normal? */
	int uNormalize;
};

struct HQFixedFunctionShaderD3D11: public HQA16ByteObject
{
	HQFixedFunctionShaderD3D11()
		: m_flags(PARAMETERS_DIRTY),
		m_constantBuffer(NULL),
		m_activeProgramIndex(0),
		m_viewMatrix(HQMatrix4::New()),
		m_projMatrix(HQMatrix4::New())
	{
		hquint32 numFFVShaders = sizeof(m_vertexShader) / sizeof(hquint32);
		hquint32 numFFPShaders = sizeof(m_pixelShader) / sizeof(hquint32);
		hquint32 numFFPrograms = sizeof(m_program) / sizeof(hquint32);

		for (hquint32 i = 0; i < numFFVShaders; ++i)
			m_vertexShader[i] = NULL;

		for (hquint32 i = 0; i < numFFPShaders; ++i)
			m_pixelShader[i] = NULL;

		for (hquint32 i = 0; i < numFFPrograms; ++i)
			m_program[i] = NULL;
	}

	~HQFixedFunctionShaderD3D11(){
		delete m_viewMatrix;
		delete m_projMatrix;
	}
	
	void SetLight(hquint32 index, const HQFFLight* light)
	{
		if (index < MAX_LIGHTS)
		{
			switch(light->type)
			{
			case HQ_LIGHT_POINT:
				memcpy(&m_parameters.uLightPosition[index], &light->position, 3 * sizeof(hqfloat32));
				m_parameters.uLightPosition[index].w = 1.0f;

				m_parameters.uLightAttenuation[index].Set(light->attenuation0, light->attenuation1, light->attenuation2);
				break;
			case HQ_LIGHT_DIRECTIONAL:
				memcpy(&m_parameters.uLightPosition[index], &light->direction, 3 * sizeof(hqfloat32));
				m_parameters.uLightPosition[index].w = 0.0f;
				break;
			}

			memcpy(&m_parameters.uLightAmbient[index], &light->ambient, 4 * sizeof(hqfloat32));
			memcpy(&m_parameters.uLightDiffuse[index], &light->diffuse, 4 * sizeof(hqfloat32));
			memcpy(&m_parameters.uLightSpecular[index], &light->specular, 4 * sizeof(hqfloat32));

			m_flags |= PARAMETERS_DIRTY;
		}
	}

	void EnableLight(hquint32 index, HQBool enable)
	{
		if (index < MAX_LIGHTS)
		{
			m_parameters.uUseLight[index] = enable;

			m_flags |= PARAMETERS_DIRTY;
		}
	}

	void SetMaterial(const HQFFMaterial* material)
	{
		memcpy(&m_parameters.uMaterialAmbient, &material->ambient, sizeof(HQColor));
		memcpy(&m_parameters.uMaterialDiffuse, &material->diffuse, sizeof(HQColor));
		memcpy(&m_parameters.uMaterialSpecular, &material->specular, sizeof(HQColor));
		memcpy(&m_parameters.uMaterialEmission, &material->emissive, sizeof(HQColor));
		m_parameters.uMaterialShininess = material->power;

		m_flags |= PARAMETERS_DIRTY;
	}

	void SetGlobalAmbient(const HQColor* color)
	{
		memcpy(&m_parameters.uAmbientColor, color, sizeof(HQColor));
		
		m_flags |= PARAMETERS_DIRTY;
	}

	//m_activeProgramIndex = {enable vertex lighting | enable specular | enable texture | enable texture}
	void EnableTexture(HQBool enable)
	{
		const hquint32 textureFlags = 1 | 1 << 1;

		if (enable)
		{

			if ((m_activeProgramIndex & textureFlags) == 0 )
			{
				m_activeProgramIndex |= textureFlags ;
				m_flags |= PROGRAM_DIRTY;
			}
		}
		else if ((m_activeProgramIndex & textureFlags) != 0)
		{
			m_activeProgramIndex &= ~textureFlags ;
			m_flags |= PROGRAM_DIRTY;
		}
	}

	void EnableLighting(HQBool enable)
	{
		const hquint32 lightingFlag = 1 << 3;

		if (enable)
		{

			if ((m_activeProgramIndex & lightingFlag) == 0 )
			{
				m_activeProgramIndex |= lightingFlag ;
				m_flags |= PROGRAM_DIRTY;
			}
		}

		else if ((m_activeProgramIndex & lightingFlag) != 0)
		{
			m_activeProgramIndex &= ~lightingFlag ;
			m_flags |= PROGRAM_DIRTY;
		}
	}

	void EnableSpecular(HQBool enable)
	{
		const hquint32 specularFlag = 1 << 2;
		if (enable)
		{

			if ((m_activeProgramIndex & specularFlag) == 0 )
			{
				m_activeProgramIndex |= specularFlag ;
				m_flags |= PROGRAM_DIRTY;
			}
		}

		else if ((m_activeProgramIndex & specularFlag) != 0)
		{
			m_activeProgramIndex &= ~specularFlag ;
			m_flags |= PROGRAM_DIRTY;
		}
	}

	hquint32 GetVertexShaderIndex(bool light, bool specular, bool texture)
	{
		return ((hquint32)light << 2) | ((hquint32)specular << 1) | (hquint32)texture;
	}

	hquint32 GetPixelShaderIndex(bool texture)
	{
		return (hquint32)texture;
	}

	hquint32 GetProgramIndex(bool light, bool specular, bool texture)
	{
		return (GetVertexShaderIndex(light, specular, texture) << 1) | GetPixelShaderIndex(texture);
	}

	HQShaderObject* & GetVertexShaderSlot(bool light, bool specular, bool texture)
	{
		return this->m_vertexShader[GetVertexShaderIndex(light, specular, texture)];
	}

	HQShaderObject* & GetPixelShaderSlot(bool texture)
	{
		return this->m_pixelShader[GetPixelShaderIndex(texture)];
	}

	HQShaderProgram* & GetProgramSlot(bool light, bool specular, bool texture)
	{
		return this->m_program[GetProgramIndex(light, specular, texture)];
	}

	void EnableNormalize(HQBool enable)
	{
		if (m_parameters.uNormalize != enable)
		{
			m_parameters.uNormalize = enable;
			m_flags |= PARAMETERS_DIRTY;
		}
	}

	void SetWorldMatrix(const HQBaseMatrix4* matrix)
	{
		memcpy(&m_parameters.uWorldMatrix, matrix, sizeof(HQMatrix4)); 
		HQMatrix4* pWorldMatrix = (HQMatrix4*) &m_parameters.uWorldMatrix;
		HQMatrix4Transpose(pWorldMatrix, pWorldMatrix);//transpose the matrix, since we will use column major in shader

		RecalculateMVP();

		m_flags |= PARAMETERS_DIRTY;
	}

	void SetViewMatrix(const HQBaseMatrix4* matrix)
	{
		memcpy(m_viewMatrix, matrix, sizeof(HQMatrix4)); 

		m_viewMatrix->Transpose();//transpose the matrix, since we will use column major in shader

		RecalculateMVP();

		//calcualte eye position from view matrix
		HQ_DECL_STACK_MATRIX4_CTOR_PARAMS(viewInv, (NULL));
		HQMatrix4Inverse(m_viewMatrix, &viewInv);

		m_parameters.uEyePos.Set(viewInv._14, viewInv._24, viewInv._34);

		m_flags |= PARAMETERS_DIRTY;
	}

	void SetProjMatrix(const HQBaseMatrix4* matrix)
	{
		memcpy(m_projMatrix, matrix, sizeof(HQMatrix4)); 

		m_projMatrix->Transpose();//transpose the matrix, since we will use column major in shader
		
		RecalculateMVP();

		m_flags |= PARAMETERS_DIRTY;
	}

	void RecalculateMVP()
	{
		HQMatrix4Multiply(m_projMatrix, m_viewMatrix, (HQMatrix4*)&m_parameters.uMvpMatrix);
		HQMatrix4Multiply((HQMatrix4*)&m_parameters.uMvpMatrix, (HQMatrix4*)&m_parameters.uWorldMatrix, (HQMatrix4*)&m_parameters.uMvpMatrix);
	}

	bool IsActive()
	{
		return (m_flags & FF_ACTIVE) != 0;
	}

	void SetActive(bool active)
	{
		if (active)
			m_flags |= FF_ACTIVE;
		else
			m_flags &= ~FF_ACTIVE;
	}

	HQShaderProgram* GetCurrentProgram()
	{
		return m_program[m_activeProgramIndex];
	}


	HQFixedFunctionParamenters m_parameters;

	HQMatrix4 *m_viewMatrix;
	HQMatrix4 *m_projMatrix;


	HQShaderObject* m_vertexShader[8];
	HQShaderObject* m_pixelShader[2];
	HQShaderProgram* m_program[16];
	HQUniformBuffer* m_constantBuffer;//const buffer
	HQSharedPtr<HQShaderConstBufferD3D11> m_constantBufferPtr;//cached pointer to const buffer
	HQSharedPtr<HQShaderConstBufferD3D11> m_prevVConstBufferPtr;//previous constant buffer set to the same vertex shader slot as this fixed function shader own buffer
	HQSharedPtr<HQShaderConstBufferD3D11> m_prevPConstBufferPtr;//previous constant buffer set to the same pixel shader slot as this fixed function shader own buffer
	hquint32 m_flags;
	hquint32 m_activeProgramIndex;
};


/*-----------------------------------------------*/


void HQShaderManagerD3D11::InitFFEmu()
{
	pFFEmu = HQ_NEW HQFixedFunctionShaderD3D11();

	/*----------create shaders---------------------------*/
#if !USE_SHADER_BYTE_CODE

#if defined _DEBUG || defined DEBUG
	HQShaderCompileMode compileMode = HQ_SCM_HLSL_10_DEBUG;
#else
	HQShaderCompileMode compileMode = HQ_SCM_HLSL_10;
#endif

	this->CreateShaderFromMemory(HQ_VERTEX_SHADER, compileMode, HQFFEmuShaderD3D1x, NULL, "VS", &pFFEmu->m_vertexShader);
	this->CreateShaderFromMemory(HQ_PIXEL_SHADER, compileMode, HQFFEmuShaderD3D1x, NULL, "PS", &pFFEmu->m_pixelShader);

#else
	this->CreateShaderFromByteCode(HQ_VERTEX_SHADER, HQFFEmuShaderD3D1x_VS_light_spec_tex, sizeof(HQFFEmuShaderD3D1x_VS_light_spec_tex),  &pFFEmu->GetVertexShaderSlot(true, true, true));
	this->CreateShaderFromByteCode(HQ_VERTEX_SHADER, HQFFEmuShaderD3D1x_VS_light_spec_notex, sizeof(HQFFEmuShaderD3D1x_VS_light_spec_notex),  &pFFEmu->GetVertexShaderSlot(true, true, false));
	this->CreateShaderFromByteCode(HQ_VERTEX_SHADER, HQFFEmuShaderD3D1x_VS_light_nospec_tex, sizeof(HQFFEmuShaderD3D1x_VS_light_nospec_tex),  &pFFEmu->GetVertexShaderSlot(true, false, true));
	this->CreateShaderFromByteCode(HQ_VERTEX_SHADER, HQFFEmuShaderD3D1x_VS_light_nospec_notex, sizeof(HQFFEmuShaderD3D1x_VS_light_nospec_notex),  &pFFEmu->GetVertexShaderSlot(true, false, false));
	this->CreateShaderFromByteCode(HQ_VERTEX_SHADER, HQFFEmuShaderD3D1x_VS_nolight_nospec_tex, sizeof(HQFFEmuShaderD3D1x_VS_nolight_nospec_tex),  &pFFEmu->GetVertexShaderSlot(false, false, true));
	this->CreateShaderFromByteCode(HQ_VERTEX_SHADER, HQFFEmuShaderD3D1x_VS_nolight_nospec_notex, sizeof(HQFFEmuShaderD3D1x_VS_nolight_nospec_notex),  &pFFEmu->GetVertexShaderSlot(false, false, false));

	this->CreateShaderFromByteCode(HQ_PIXEL_SHADER, HQFFEmuShaderD3D1x_PS_tex, sizeof(HQFFEmuShaderD3D1x_PS_tex),  &pFFEmu->GetPixelShaderSlot(true));
	this->CreateShaderFromByteCode(HQ_PIXEL_SHADER, HQFFEmuShaderD3D1x_PS_notex, sizeof(HQFFEmuShaderD3D1x_PS_notex),  &pFFEmu->GetPixelShaderSlot(false));

#endif
	
	for (int light = 0; light < 2; ++light)
	{
		for (int specular = 0; specular < 2; ++specular)
		{
			if (light == 0 && specular == 1)//specular enable has no meaning when lighting is disable
				continue;
			for (int texture = 0; texture < 2; ++texture)
			{
#if defined WIN32 || defined _MSC_VER
#	pragma warning(push)
#	pragma warning(disable:4800)
#endif

				bool useLighting = (bool) light;
				bool useSpecular = (bool)specular;
				bool useTexture = (bool) texture;

#if defined WIN32 || defined _MSC_VER
#	pragma warning(pop)
#endif


				this->CreateProgram(
					pFFEmu->GetVertexShaderSlot(useLighting, useSpecular,  useTexture),
					pFFEmu->GetPixelShaderSlot(useTexture), 
					NULL, 
					&pFFEmu->GetProgramSlot(useLighting, useSpecular,  useTexture));
			}//for (int texture = 0; texture < 2; ++texture)
		}//for (int specular = 0; specular < 2; ++specular)
	}//for (int light = 0; light < 2; ++light)

	/*---------create constant buffer--------------------*/

	this->CreateUniformBuffer(NULL, sizeof(HQFixedFunctionParamenters), false, &pFFEmu->m_constantBuffer);
	
	pFFEmu->m_constantBufferPtr = this->shaderConstBuffers.GetItemPointer(pFFEmu->m_constantBuffer);//cache the pointer

	/*---------set default values------------------------*/

	//default light
	HQFFLight defaultLight;
	HQBool falseVal = HQ_FALSE;
	HQBool trueVal = HQ_TRUE;

	for (hquint32 i = 0; i < MAX_LIGHTS; ++i)
	{
		SetFFRenderState((HQFFRenderState)(HQ_LIGHT0 + i) , &defaultLight);
		SetFFRenderState((HQFFRenderState)(HQ_LIGHT0_ENABLE + i), &falseVal);//disable light i
	}

	SetFFRenderState(HQ_TEXTURE_ENABLE, &trueVal);//enable texture

	SetFFRenderState(HQ_LIGHTING_ENABLE, &falseVal);//disable lighting

	SetFFRenderState(HQ_SPECULAR_ENABLE, &falseVal);//disable specular

	SetFFRenderState(HQ_NORMALIZE_NORMALS, &falseVal);//disable normal auto normalization

	HQColorui black = HQColoruiRGBA(0, 0, 0, 255, CL_BGRA);

	SetFFRenderState(HQ_AMBIENT, &black);//set black global ambient color

	ActiveFFEmu();//active fixed function shader
}

void HQShaderManagerD3D11::ReleaseFFEmu()
{
	//no need to delete shaders and buffers because they will be deleted by ShaderManager
	delete pFFEmu;
}

HQReturnVal HQShaderManagerD3D11::ActiveFFEmu()
{
	if (!pFFEmu->IsActive())
	{
		pFFEmu->m_prevVConstBufferPtr = this->uBufferSlots[0][0];//save previous set vertex shader buffer
		pFFEmu->m_prevPConstBufferPtr = this->uBufferSlots[2][0];//save previous set pixel shader buffer

		this->ActiveProgram(pFFEmu->GetCurrentProgram());//active the fixed function program

		pFFEmu->m_flags &= ~PROGRAM_DIRTY;

		this->SetUniformBuffer(HQ_VERTEX_SHADER| 0, pFFEmu->m_constantBuffer);//set the fixed function shader's own buffer
		this->SetUniformBuffer(HQ_PIXEL_SHADER| 0, pFFEmu->m_constantBuffer);//set the fixed function shader's own buffer

		pFFEmu->SetActive(true);
	}
	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D11::DeActiveFFEmu()
{
	if (pFFEmu->IsActive())
	{
		if ((pFFEmu->m_flags & PROGRAM_SWITCHING) == 0)//if it is not zero then this method must be called when the fixed function controller is trying to switch the shader program
		{
			//vertex shader buffer
			if (this->uBufferSlots[0][0] != pFFEmu->m_constantBufferPtr)//this may be a change outside since last caching
			{
				pFFEmu->m_prevVConstBufferPtr = this->uBufferSlots[0][0];//save vertex shader's previous bound const buffer
			}
			else {
				//restore previous buffer
				if (pFFEmu->m_prevVConstBufferPtr != NULL)
					pD3DContext->VSSetConstantBuffers(0, 1, &pFFEmu->m_prevVConstBufferPtr->pD3DBuffer);

				this->uBufferSlots[0][0] = pFFEmu->m_prevVConstBufferPtr;
			}

			//pixel shader buffer
			if (this->uBufferSlots[2][0] != pFFEmu->m_constantBufferPtr)//this may be a change outside since last caching
			{
				pFFEmu->m_prevPConstBufferPtr = this->uBufferSlots[2][0];//save pixel shader's previous bound const buffer

			}
			else
			{   //restore previous buffer
				if (pFFEmu->m_prevPConstBufferPtr != NULL)
					pD3DContext->PSSetConstantBuffers(0, 1, &pFFEmu->m_prevPConstBufferPtr->pD3DBuffer);

				this->uBufferSlots[2][0] = pFFEmu->m_prevPConstBufferPtr;
			}

			pFFEmu->SetActive(false);
		}
	}
	return HQ_OK;
}

bool HQShaderManagerD3D11::IsFFEmuActive()
{
	return pFFEmu->IsActive();
}

HQShaderObject* HQShaderManagerD3D11::GetFFVertexShaderForInputLayoutCreation()
{
	return pFFEmu->GetVertexShaderSlot(false, false, true);
}

bool HQShaderManagerD3D11::IsFFShader(HQShaderObject* shaderID)//fixed function pixel shader
{
	for (int light = 0; light < 2; ++light)
	{
		for (int specular = 0; specular < 2; ++specular)
		{
			if (light == 0 && specular == 1)//specular enable has no meaning when lighting is disable
				continue;
			for (int texture = 0; texture < 2; ++texture)
			{
#if defined WIN32 || defined _MSC_VER
#	pragma warning(push)
#	pragma warning(disable:4800)
#endif

				bool useLighting = (bool) light;
				bool useSpecular = (bool)specular;
				bool useTexture = (bool) texture;

#if defined WIN32 || defined _MSC_VER
#	pragma warning(pop)
#endif


				if (shaderID == pFFEmu->GetVertexShaderSlot(useLighting, useSpecular,  useTexture))
					return true;

				if (shaderID == pFFEmu->GetPixelShaderSlot(useTexture))
					return true;
			}//for (int texture = 0; texture < 2; ++texture)
		}//for (int specular = 0; specular < 2; ++specular)
	}//for (int light = 0; light < 2; ++light)

	return false;
}

bool HQShaderManagerD3D11::IsFFProgram(HQShaderProgram* programID)//fixed function program
{
	for (int light = 0; light < 2; ++light)
	{
		for (int specular = 0; specular < 2; ++specular)
		{
			if (light == 0 && specular == 1)//specular enable has no meaning when lighting is disable
				continue;
			for (int texture = 0; texture < 2; ++texture)
			{
#if defined WIN32 || defined _MSC_VER
#	pragma warning(push)
#	pragma warning(disable:4800)
#endif

				bool useLighting = (bool) light;
				bool useSpecular = (bool)specular;
				bool useTexture = (bool) texture;

#if defined WIN32 || defined _MSC_VER
#	pragma warning(pop)
#endif


				if (programID == pFFEmu->GetProgramSlot(useLighting, useSpecular,  useTexture))
					return true;
			}//for (int texture = 0; texture < 2; ++texture)
		}//for (int specular = 0; specular < 2; ++specular)
	}//for (int light = 0; light < 2; ++light)

	return false;
}

bool HQShaderManagerD3D11::IsFFConstBuffer(HQUniformBuffer* id)//fixed function const buffer
{
	return id == pFFEmu->m_constantBuffer;
}

HQReturnVal HQShaderManagerD3D11::SetFFRenderState(HQFFRenderState stateType, const void* pValue)
{
	switch(stateType)
	{
	case HQ_LIGHT0: case HQ_LIGHT1: case HQ_LIGHT2:
	case HQ_LIGHT3: 
		pFFEmu->SetLight(stateType , (HQFFLight *) pValue);
		break;
	case HQ_LIGHT0_ENABLE : case HQ_LIGHT1_ENABLE: case HQ_LIGHT2_ENABLE: 
	case HQ_LIGHT3_ENABLE: 
		pFFEmu-> EnableLight(stateType - HQ_LIGHT0_ENABLE, *(HQBool*)pValue);
		break;
	case HQ_MATERIAL:
		pFFEmu->SetMaterial((HQFFMaterial *)pValue);
		break;
	case HQ_TEXTURE_ENABLE:
		pFFEmu->EnableTexture(*(HQBool*)pValue);
		break;
	case HQ_AMBIENT:
		{
			//color is BGRA order
			HQColorui color = *((HQColorui*)pValue);
			HQColor colorF;
			colorF.r = ((color >> 16)  & 0xff) / 255.f;
			colorF.g = ((color >> 8) & 0xff) / 255.f;
			colorF.b = ((color >> 0) & 0xff) / 255.f;
			colorF.a = ((color >> 24) & 0xff) / 255.f;

			pFFEmu->SetGlobalAmbient(&colorF);
		}
		break;
	case HQ_LIGHTING_ENABLE:
		pFFEmu->EnableLighting(*(HQBool*)pValue);
		break;
	case HQ_SPECULAR_ENABLE:
		pFFEmu->EnableSpecular(*(HQBool*)pValue);
		break;
	case HQ_NORMALIZE_NORMALS:
		pFFEmu->EnableNormalize(*(HQBool*)pValue);
		break;
	default:
		return HQ_FAILED;
	}//switch(stateType)

	return HQ_OK;
}

HQReturnVal HQShaderManagerD3D11::SetFFTransform(HQFFTransformMatrix type, const HQBaseMatrix4 *pMatrix)
{
	switch (type)
	{
	case HQ_WORLD:
		pFFEmu->SetWorldMatrix(pMatrix);
		break;
	case HQ_VIEW:
		pFFEmu->SetViewMatrix(pMatrix);
		break;
	case HQ_PROJECTION:
		pFFEmu->SetProjMatrix(pMatrix);
		break;

	}

	return HQ_OK;
}

void HQShaderManagerD3D11::NotifyFFRenderIfNeeded()// notify shader manager that the render device is going to draw something. Shader manager needs to update Fixed Function emulator if needed
{
	if (IsFFEmuActive() == false)
		return;//no thing to do

	//vertex shader buffer
	if (this->uBufferSlots[0][0] != pFFEmu->m_constantBufferPtr)//this may be a change outside since last draw call
	{
		pFFEmu->m_prevVConstBufferPtr = this->uBufferSlots[0][0];//save vertex shader's previous bound const buffer

		this->SetUniformBuffer(HQ_VERTEX_SHADER| 0, pFFEmu->m_constantBuffer);//set the fixed function shader's own buffer
	}

	//pixel shader buffer
	if (this->uBufferSlots[2][0] != pFFEmu->m_constantBufferPtr)//this may be a change outside since last draw call
	{
		pFFEmu->m_prevPConstBufferPtr = this->uBufferSlots[2][0];//save pixel shader's previous bound const buffer

		this->SetUniformBuffer(HQ_PIXEL_SHADER| 0, pFFEmu->m_constantBuffer);//set the fixed function shader's own buffer
	}

	//check if const buffer needs to be updated
	if (pFFEmu->m_flags & PARAMETERS_DIRTY)
	{
		
		pFFEmu->m_constantBuffer->Update( &pFFEmu->m_parameters);//update buffer

		pFFEmu->m_flags &= ~PARAMETERS_DIRTY;
	}

	//check if we need to switch program
	if (pFFEmu->m_flags & PROGRAM_DIRTY)
	{
		pFFEmu->m_flags |= PROGRAM_SWITCHING;//prevent the next method call from deactivating the fixed function shader

		this->ActiveProgram(pFFEmu->GetCurrentProgram());//active the fixed function program

		pFFEmu->m_flags &= ~PROGRAM_DIRTY;

		pFFEmu->m_flags &= ~PROGRAM_SWITCHING;
	}

}
