/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceGLPCH.h"

#include "HQShaderGL.h"
#include "HQShaderGL_GLSL_VarParser.h"
#include "HQVertexStreamManagerGL.h"
#include "HQDeviceGL.h"

#include "../BaseImpl/BaseImplShaderString/HQFFEmuShaderGL.h"

#ifdef OPTIMIZE_GLSL
#include <string>
#include <sstream>

#include "glsl_optimizer.h"
#endif

#define MAX_LIGHTS 4

#define PARAMETERS_DIRTY_MASK 0x0000ffff
#define PROGRAM_DIRTY 0x2
#define FF_ACTIVE		  0x4

#define GET_UNIFORM_LOC(dicPtr, program, name) {dicPtr->name##Loc = glGetUniformLocation(program, #name);}

#define GET_UNIFORM_LIGHT_LOC(dicPtr, program, lightAttrib, index) \
	{\
		lightAttrib##Name[strlen(#lightAttrib) + 1] = '0' + index;\
		dicPtr->lightAttrib##Loc[index] = glGetUniformLocation(program, lightAttrib##Name);\
	}

struct HQFixedFunctionParamenters: public HQA16ByteObject
{
	HQFixedFunctionParamenters()
	{
		memset(this, 0, sizeof(HQFixedFunctionParamenters));

		uMvpMatrix = HQMatrix4::New();
		uWorldMatrix = HQMatrix4::New();
	}

	~HQFixedFunctionParamenters()
	{
		delete uMvpMatrix;
		delete uWorldMatrix;
	}

	/*------------uniform values-------------*/
	/* Matrix Uniforms */

	HQMatrix4* uMvpMatrix;
	HQMatrix4* uWorldMatrix;

	/* Light Uniforms */
	HQFloat4  uLightPosition[MAX_LIGHTS] ;
	HQFloat4  uLightAmbient[MAX_LIGHTS];
	HQFloat4  uLightDiffuse[MAX_LIGHTS] ;
	HQFloat4  uLightSpecular[MAX_LIGHTS] ;
	HQFloat4  uLightAttenuation[MAX_LIGHTS];// w is ignored. C struct need to add padding float at the end of each element
	
	/* Global ambient color */
	HQColor  uAmbientColor;

	/* Material Uniforms */
	HQColor  uMaterialAmbient;
	HQColor  uMaterialEmission;
	HQColor  uMaterialDiffuse;
	HQColor  uMaterialSpecular;
	float uMaterialShininess;
	
	/* Eye position */
	HQFloat3 uEyePos;

	/* For light 0 - 3 */
	float  uUseLight[4];

	/* Normalize normal? */
	float uNormalize;

	/*------uniform dirty masks--------------*/
	/* Matrix Uniforms */

	hquint32 uMvpMatrixMask;
	hquint32 uWorldMatrixMask;

	/* Light Uniforms */
	hquint32  uLightMask    [MAX_LIGHTS];

	/* Global ambient color */
	hquint32  uAmbientColorMask;

	/* Material Uniforms */
	hquint32  uMaterialMask;
	
	/* Eye position */
	hquint32 uEyePosMask;

	/* For light 0 - 3 */
	hquint32  uUseLightMask;

	/* Normalize normal? */
	hquint32 uNormalizeMask;
};

struct HQFixedFunctionParametersLocGL
{
	/*------uniform locations--------------*/
	/* Matrix Uniforms */

	GLint uMvpMatrixLoc;
	GLint uWorldMatrixLoc;

	/* Light Uniforms */
	GLint  uLightPositionLoc    [MAX_LIGHTS];
	GLint  uLightAmbientLoc     [MAX_LIGHTS];
	GLint  uLightDiffuseLoc     [MAX_LIGHTS];
	GLint  uLightSpecularLoc    [MAX_LIGHTS];
	GLint  uLightAttenuationLoc [MAX_LIGHTS];// w is ignored. C struct need to add padding float at the end of each element
	
	/* Global ambient color */
	GLint  uAmbientColorLoc;

	/* Material Uniforms */
	GLint  uMaterialAmbientLoc;
	GLint  uMaterialEmissionLoc;
	GLint  uMaterialDiffuseLoc;
	GLint  uMaterialSpecularLoc;
	GLint uMaterialShininessLoc;
	
	/* Eye position */
	GLint uEyePosLoc;

	/* For light 0 - 3 */
	GLint  uUseLightLoc;

	/* Normalize normal? */
	GLint uNormalizeLoc;
};

struct HQFixedFunctionShaderGL: public HQA16ByteObject
{
	HQFixedFunctionShaderGL()
		: m_flags(0),
		m_activeProgramIndex(0), 
		m_viewMatrix (HQMatrix4::New()),
		m_projMatrix (HQMatrix4::New())
	{
		m_parameters.uNormalizeMask = PARAMETERS_DIRTY_MASK;//force resync normalize uniform on next shader activation

		const hquint32 numFFVShaders = sizeof(m_vertexShader) / sizeof(hquint32);
		const hquint32 numFFPShaders = sizeof(m_pixelShader) / sizeof(hquint32);
		const hquint32 numFFPrograms = sizeof(m_program) / sizeof(hquint32);

		for (hquint32 i = 0; i < numFFVShaders; ++i)
			m_vertexShader[i] = 0;

		for (hquint32 i = 0; i < numFFPShaders; ++i)
			m_pixelShader[i] = 0;

		for (hquint32 i = 0; i < numFFPrograms; ++i)
		{
			m_program[i] = 0;
			m_uniformLoc[i] = NULL;
		}
		
		Restore();
	}
	
	~HQFixedFunctionShaderGL()
	{
		const hquint32 numFFPrograms = sizeof(m_program) / sizeof(hquint32);
		
		for (hquint32 i = 0; i < numFFPrograms; ++i)
		{
			if (m_uniformLoc[i] != 0)
			{
				delete m_uniformLoc[i];
			}
		}

		Release();

		delete m_viewMatrix;
		delete m_projMatrix;
	}

	void Release()
	{
		const hquint32 numFFVShaders = sizeof(m_vertexShader) / sizeof(hquint32);
		const hquint32 numFFPShaders = sizeof(m_pixelShader) / sizeof(hquint32);
		const hquint32 numFFPrograms = sizeof(m_program) / sizeof(hquint32);


		for (hquint32 i = 0; i < numFFPrograms; ++i)
		{
			if (m_program[i] != 0)
			{
				glDeleteProgram(m_program[i]);
				m_program[i] = 0;
			}
		}

		for (hquint32 i = 0; i < numFFVShaders; ++i)
		{
			if (m_vertexShader[i] != 0)
			{
				glDeleteShader(m_vertexShader[i]);
				m_vertexShader[i] = 0;
			}
		}

		for (hquint32 i = 0; i < numFFPShaders; ++i)
		{
			if (m_pixelShader[i] != 0)
			{
				glDeleteShader(m_pixelShader[i]);
				m_pixelShader[i] = 0;
			}
		}

		//mark all parameters dirty
		m_parameters.uMvpMatrixMask = 
		m_parameters.uWorldMatrixMask = 
		m_parameters.uAmbientColorMask =
		m_parameters.uMaterialMask = 
		m_parameters.uEyePosMask = 
		m_parameters.uUseLightMask = 
		m_parameters.uNormalizeMask = PARAMETERS_DIRTY_MASK;

		for (int i = 0; i < MAX_LIGHTS; ++i)
		{
			m_parameters.uLightMask[i] = PARAMETERS_DIRTY_MASK;
		}

		//force re-active shader next draw call
		m_flags |= PROGRAM_DIRTY;

	}

	void Restore()
	{
		CreateProgram(true, true, true);
		CreateProgram(true, true, false);
		CreateProgram(true, false, true);
		CreateProgram(true, false, false);
		CreateProgram(false, false, true);
		CreateProgram(false, false, false);
	}

private:

#ifdef OPTIMIZE_GLSL
	bool optimizeGLSL(GLenum glshaderType, const char **sourceArray, int sourceArrayLen, std::string &optimized_source){
		//combine source array
		std::stringstream ss ;
		for (int i = 0; i < sourceArrayLen; ++i){
			ss << sourceArray[i];
		}

		std::string source = ss.str();

		glslopt_ctx*  glsl_optContext = glslopt_initialize(kGlslTargetOpenGL);

		glslopt_shader_type shaderType = glshaderType == GL_VERTEX_SHADER? kGlslOptShaderVertex : kGlslOptShaderFragment;

		bool success = true;
		glslopt_shader* shader = glslopt_optimize(glsl_optContext, shaderType, source.c_str(), 0);

		if(glslopt_get_status(shader))
		{
			optimized_source = glslopt_get_output(shader);
		}
		else
		{
			const char * log = glslopt_get_log(shader);//for debugging
			success = false;

		}
		glslopt_shader_delete(shader);

		glslopt_cleanup(glsl_optContext);

		return success;
	}
#endif

	HQReturnVal CreateShader(GLenum shaderType,
								  const char** sourceArray,
								  GLuint sourceSize,
								  GLuint & shaderOut)
	{
#ifdef OPTIMIZE_GLSL
		//optimize it before sending to opengl
		std::string opt_code;
		if (optimizeGLSL(shaderType, sourceArray, sourceSize, opt_code))
		{
			*sourceArray = opt_code.c_str();
			sourceSize = 1;
		}
#endif

		shaderOut = glCreateShader(shaderType);

		glShaderSource(shaderOut, sourceSize, (const GLchar**)sourceArray, NULL);
		glCompileShader(shaderOut);

		GLint compileOK;
		glGetShaderiv(shaderOut , GL_COMPILE_STATUS , &compileOK);
		if(shaderOut==0 || compileOK == GL_FALSE)
		{
			int infologLength = 0;
			int charsWritten  = 0;
			char *infoLog;
			glGetShaderiv(shaderOut, GL_INFO_LOG_LENGTH,&infologLength);
			if (infologLength > 0)
			{
				infoLog = (char *)malloc(infologLength);
				glGetShaderInfoLog(shaderOut, infologLength, &charsWritten, infoLog);
				g_pShaderMan->Log("GLSL fixed function shader compile error: %s",infoLog);
				free(infoLog);
			}
			else
				g_pShaderMan->Log("GLSL fixed function shader compile error");
			return HQ_FAILED;
		}

		return HQ_OK;
	}


	void CreateProgram(bool light, bool specular, bool texture)
	{
		GLuint & vshader = GetVertexShaderSlot(light, specular, texture);
		GLuint & pshader = GetPixelShaderSlot(texture);

		hquint32 programIndex = GetProgramIndex(light, specular, texture);
		GLuint & program = m_program[programIndex];

		const char lightingDefine[] = "#define USE_LIGHTING\n";
		const char specularDefine[] = "#define USE_SPECULAR\n";
		const char textureDefine[] = "#define USE_TEXTURE\n";

		size_t sourceArrayCount;

		char version_line[] = "#version 110\n#define highp\n#define lowp\n#define mediump\n";

		if (GLEW_VERSION_3_0)
			strncpy(version_line, "#version 130", 12);

		//create vertex shader
		if (vshader == 0)//only create when it is not created before
		{
			const GLchar* vshaderSourceArray[] = {
#ifdef HQ_OPENGLES
				"#version 100\n"
#else
				version_line,
#endif
				"#define VERTEX_SHADER\n",
				light? lightingDefine: "",
				specular? specularDefine: "",
				texture? textureDefine: "",

				HQFFEmuShaderGL
			};

			sourceArrayCount = sizeof(vshaderSourceArray) / sizeof(GLchar *);
			
			g_pShaderMan->Log("creating GLSL fixed function vertex shader(light=%d, specular=%d, texture=%d)", (int)light, (int)specular, (int)texture);

			CreateShader(GL_VERTEX_SHADER, vshaderSourceArray, sourceArrayCount, vshader);
		}
		//create fragment shader

		if (pshader == 0)//only create when it is not created before
		{
			const GLchar* fshaderSourceArray[] = {
#ifdef HQ_OPENGLES
				"#version 100\n"
#else
				version_line,
#endif
				"#define FRAGMENT_SHADER\n",
				texture? textureDefine: "",

				HQFFEmuShaderGL
			};

			sourceArrayCount = sizeof(fshaderSourceArray) / sizeof(GLchar *);
			
			g_pShaderMan->Log("creating GLSL fixed function fragment shader(texture=%d)", (int)texture);

			CreateShader(GL_FRAGMENT_SHADER, fshaderSourceArray, sourceArrayCount, pshader);
		}

		//create program
		g_pShaderMan->Log("creating GLSL fixed function program(light=%d, specular=%d, texture=%d): vshader=%u, fshader=%u", (int)light, (int)specular, (int)texture, vshader, pshader);

		program = glCreateProgram();

		glAttachShader(program, vshader);
		glAttachShader(program, pshader);

		glLinkProgram(program);

		//bind attribute locations
		glBindAttribLocation(program, POS_ATTRIB, "aPosition");
		glBindAttribLocation(program, COLOR_ATTRIB, "aColor");
		glBindAttribLocation(program, NORMAL_ATTRIB, "aNormal");
		glBindAttribLocation(program, TEXCOORD0_ATTRIB, "aTexcoords");

		glLinkProgram(program);//re-link

		GLint OK;
		glGetProgramiv(program , GL_LINK_STATUS , &OK);

		if(OK == GL_FALSE)
		{
			int infologLength = 0;
			int charsWritten  = 0;
			char *infoLog;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH,&infologLength);
			if (infologLength > 0)
			{
				infoLog = (char *)malloc(infologLength);
				glGetProgramInfoLog(program, infologLength, &charsWritten, infoLog);
				g_pShaderMan->Log("GLSL fixed function program(light=%d, specular=%d, texture=%d) link error: %s", (int)light, (int)specular, (int)texture, infoLog);
				free(infoLog);
			}
			return ;
		}

		//bind the texture to texture unit 0
		GLint uniformTextureLoc = glGetUniformLocation(program, "texture0");
		if (uniformTextureLoc >= 0)
		{
			GLint oldProgram = 0;
			glGetIntegerv(GL_CURRENT_PROGRAM, &oldProgram);
			
			glUseProgram(program);
			glUniform1i(uniformTextureLoc, 0);
			glUseProgram(oldProgram);
		}

		//check if this program is valid
		glValidateProgram(program);
		glGetProgramiv(program, GL_VALIDATE_STATUS, &OK);
		if(OK == GL_FALSE)
		{
			int infologLength = 0;
			int charsWritten  = 0;
			char *infoLog;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH,&infologLength);
			if (infologLength > 0)
			{
				infoLog = (char *)malloc(infologLength);
				glGetProgramInfoLog(program, infologLength, &charsWritten, infoLog);
				g_pShaderMan->Log("GLSL fixed function program(light=%d, specular=%d, texture=%d) validate error: %s", (int)light, (int)specular, (int)texture, infoLog);
				free(infoLog);
			}
			return ;
		}

		//everything is ok, now get the uniform locations
		GetUniformLoc(programIndex);
	}

	void GetUniformLoc(hquint32 programIndex)
	{
		GLuint program = m_program[programIndex];
		
		//create new uniform locations dictionary for this program
		HQFixedFunctionParametersLocGL *& uniformLocations = m_uniformLoc[programIndex];
		if (uniformLocations == NULL)
			uniformLocations = HQ_NEW HQFixedFunctionParametersLocGL();

		//get uniform locations
		GET_UNIFORM_LOC(uniformLocations, program, uMvpMatrix);
		GET_UNIFORM_LOC(uniformLocations, program, uWorldMatrix);

		{
			char uLightPositionName[] = "uLightPosition[0]";
			char uLightAmbientName[] = "uLightAmbient[0]";
			char uLightDiffuseName[] = "uLightDiffuse[0]";
			char uLightSpecularName[] = "uLightSpecular[0]";
			char uLightAttenuationName[] = "uLightAttenuation[0]";

			for (int i = 0; i < MAX_LIGHTS; ++i)
			{
				GET_UNIFORM_LIGHT_LOC(uniformLocations, program, uLightPosition, i);
				GET_UNIFORM_LIGHT_LOC(uniformLocations, program, uLightAmbient, i);
				GET_UNIFORM_LIGHT_LOC(uniformLocations, program, uLightDiffuse, i);
				GET_UNIFORM_LIGHT_LOC(uniformLocations, program, uLightSpecular, i);
				GET_UNIFORM_LIGHT_LOC(uniformLocations, program, uLightAttenuation, i);
			}
		}

		GET_UNIFORM_LOC(uniformLocations, program, uAmbientColor);
		GET_UNIFORM_LOC(uniformLocations, program, uMaterialAmbient);
		GET_UNIFORM_LOC(uniformLocations, program, uMaterialEmission);
		GET_UNIFORM_LOC(uniformLocations, program, uMaterialDiffuse);
		GET_UNIFORM_LOC(uniformLocations, program, uMaterialSpecular);
		GET_UNIFORM_LOC(uniformLocations, program, uMaterialShininess);
		GET_UNIFORM_LOC(uniformLocations, program, uEyePos);
		GET_UNIFORM_LOC(uniformLocations, program, uUseLight);
		GET_UNIFORM_LOC(uniformLocations, program, uNormalize);

	}


public:
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

			m_parameters.uLightMask[index] = PARAMETERS_DIRTY_MASK;
		}
	}

	void EnableLight(hquint32 index, HQBool enable)
	{
		if (index < MAX_LIGHTS)
		{
			m_parameters.uUseLight[index] = (float)enable;

			m_parameters.uUseLightMask = PARAMETERS_DIRTY_MASK;
		}
	}

	void SetMaterial(const HQFFMaterial* material)
	{
		memcpy(&m_parameters.uMaterialAmbient, &material->ambient, sizeof(HQColor));
		memcpy(&m_parameters.uMaterialDiffuse, &material->diffuse, sizeof(HQColor));
		memcpy(&m_parameters.uMaterialSpecular, &material->specular, sizeof(HQColor));
		memcpy(&m_parameters.uMaterialEmission, &material->emissive, sizeof(HQColor));
		m_parameters.uMaterialShininess = material->power;

		m_parameters.uMaterialMask = PARAMETERS_DIRTY_MASK;
	}

	void SetGlobalAmbient(const HQColor* color)
	{
		memcpy(&m_parameters.uAmbientColor, color, sizeof(HQColor));
		
		m_parameters.uAmbientColorMask = PARAMETERS_DIRTY_MASK;
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


	GLuint & GetVertexShaderSlot(bool light, bool specular, bool texture)
	{
		return this->m_vertexShader[GetVertexShaderIndex(light, specular, texture)];
	}

	bool DoesCurrentProgramUseLighting()
	{
		const hquint32 lightingFlag = 1 << 3;

		return (m_activeProgramIndex & lightingFlag) != 0;
	}

	bool DoesCurrentProgramUseSpecular()
	{
		const hquint32 specularFlag = 1 << 2;

		return (m_activeProgramIndex & specularFlag) != 0;
	}

	GLuint & GetPixelShaderSlot( bool texture)
	{
		return this->m_pixelShader[GetPixelShaderIndex(texture)];
	}

	GLuint & GetProgramSlot(bool light, bool specular, bool texture)
	{
		return this->m_program[GetProgramIndex(light, specular, texture)];
	}

	hquint32 GetActiveProgramShiftMask()
	{
		return 0x1 << m_activeProgramIndex;
	}

	void EnableNormalize(HQBool enable)
	{
		if (m_parameters.uNormalize != (float)enable)
		{
			m_parameters.uNormalize = (float)enable;
			m_parameters.uNormalizeMask = PARAMETERS_DIRTY_MASK;
		}
	}

	void SetWorldMatrix(const HQBaseMatrix4* matrix)
	{
		memcpy(m_parameters.uWorldMatrix, matrix, sizeof(HQMatrix4)); 
		m_parameters.uWorldMatrix->Transpose();//transpose the matrix, since we will use column major in shader

		RecalculateMVP();

		m_parameters.uWorldMatrixMask = PARAMETERS_DIRTY_MASK;
	}

	void SetViewMatrix(const HQBaseMatrix4* matrix)
	{
		memcpy(m_viewMatrix, matrix, sizeof(HQMatrix4)); 

		m_viewMatrix->Transpose();//transpose the matrix, since we will use column major in shader

		RecalculateMVP();

		//calcualte eye position from view matrix
		HQ_DECL_STACK_MATRIX4_CTOR_PARAMS( viewInv, (NULL));
		HQMatrix4Inverse(m_viewMatrix, &viewInv);

		m_parameters.uEyePos.Set(viewInv._14, viewInv._24, viewInv._34);
		m_parameters.uEyePosMask = PARAMETERS_DIRTY_MASK;

	}

	void SetProjMatrix(const HQBaseMatrix4* matrix)
	{
		memcpy(m_projMatrix, matrix, sizeof(HQMatrix4)); 

		m_projMatrix->Transpose();//transpose the matrix, since we will use column major in shader
		
		RecalculateMVP();
	}

	void RecalculateMVP()
	{
		HQMatrix4Multiply(m_projMatrix, m_viewMatrix, m_parameters.uMvpMatrix);
		HQMatrix4Multiply(m_parameters.uMvpMatrix, m_parameters.uWorldMatrix, m_parameters.uMvpMatrix);

		m_parameters.uMvpMatrixMask = PARAMETERS_DIRTY_MASK;
	}

	bool IsActive()
	{
		return (m_flags & FF_ACTIVE) != 0;
	}

	void SetActive(bool active)
	{
		if (active)
		{
			glUseProgram(GetCurrentProgram());

			m_flags &= ~PROGRAM_DIRTY;//don't need to switch if there is not any change in lighting, specular and texture

			m_flags |= FF_ACTIVE;
		}
		else
			m_flags &= ~FF_ACTIVE;
	}

	GLuint GetCurrentProgram()
	{
		return m_program[m_activeProgramIndex];
	}

	HQFixedFunctionParametersLocGL * GetCurrentProgramUniformLoc()
	{
		return m_uniformLoc[m_activeProgramIndex];
	}

	//commit uniform change to the program, and/or change the program if needed
	void CommitChange()
	{
		if (m_flags & PROGRAM_DIRTY)
		{
			glUseProgram(GetCurrentProgram());

			m_flags &= ~PROGRAM_DIRTY;
		}

		//check if there are any uniform changes 
		const hquint32 currentProgramMask = GetActiveProgramShiftMask();
		const hquint32 negCurrentProgramMask = ~currentProgramMask;

		HQFixedFunctionParametersLocGL * uniforms = GetCurrentProgramUniformLoc();

		//mvp matrix
		if (m_parameters.uMvpMatrixMask & currentProgramMask)
		{
			glUniformMatrix4fv(uniforms->uMvpMatrixLoc, 1, GL_FALSE, (GLfloat*)m_parameters.uMvpMatrix);

			m_parameters.uMvpMatrixMask &= negCurrentProgramMask;
		}

		//world matrix
		if (m_parameters.uWorldMatrixMask & currentProgramMask)
		{
			glUniformMatrix4fv(uniforms->uWorldMatrixLoc, 1, GL_FALSE, (GLfloat*)m_parameters.uWorldMatrix);

			m_parameters.uWorldMatrixMask &= negCurrentProgramMask;
		}

		//lighting
		if (DoesCurrentProgramUseLighting())
		{
			bool programUseSpecular = DoesCurrentProgramUseSpecular();
			//eye position
			if (programUseSpecular && (m_parameters.uEyePosMask & currentProgramMask) != 0)
			{
				//eye position is used when specular is enabled
				glUniform3fv(uniforms->uEyePosLoc, 1, m_parameters.uEyePos.f);

				m_parameters.uEyePosMask &= negCurrentProgramMask;
			}

			//ambient color 
			if (m_parameters.uAmbientColorMask & currentProgramMask)
			{
				glUniform4fv(uniforms->uAmbientColorLoc, 1, m_parameters.uAmbientColor);

				m_parameters.uAmbientColorMask &= negCurrentProgramMask;
			}

			//normalize normal
			if (m_parameters.uNormalizeMask & currentProgramMask)
			{
				glUniform1f(uniforms->uNormalizeLoc, m_parameters.uNormalize);

				m_parameters.uNormalizeMask &= negCurrentProgramMask;
			}

			//material
			if (m_parameters.uMaterialMask & currentProgramMask)
			{
				glUniform4fv(uniforms->uMaterialAmbientLoc, 1, m_parameters.uMaterialAmbient);
				glUniform4fv(uniforms->uMaterialEmissionLoc, 1, m_parameters.uMaterialEmission);
				glUniform4fv(uniforms->uMaterialDiffuseLoc, 1, m_parameters.uMaterialDiffuse);
				if (programUseSpecular)
				{
					//only sync when program use specular
					glUniform4fv(uniforms->uMaterialSpecularLoc, 1, m_parameters.uMaterialSpecular);
					glUniform1f(uniforms->uMaterialShininessLoc, m_parameters.uMaterialShininess);
				}

				m_parameters.uMaterialMask &= negCurrentProgramMask;
			}

			//light(i) enable
			if (m_parameters.uUseLightMask & currentProgramMask)
			{
				glUniform4fv(uniforms->uUseLightLoc, 1, m_parameters.uUseLight);

				m_parameters.uUseLightMask &= negCurrentProgramMask;
			}

			//light attributes
			for (int i = 0; i < MAX_LIGHTS; ++i)
			{
				if (m_parameters.uLightMask[i] & currentProgramMask)
				{
					//light [i] has been changed

					glUniform4fv(uniforms->uLightPositionLoc[i], 1, m_parameters.uLightPosition[i].f);
					glUniform4fv(uniforms->uLightAmbientLoc[i], 1, m_parameters.uLightAmbient[i].f);
					glUniform4fv(uniforms->uLightDiffuseLoc[i], 1, m_parameters.uLightDiffuse[i].f);
					glUniform4fv(uniforms->uLightAttenuationLoc[i], 1, m_parameters.uLightAttenuation[i].f);
					if (programUseSpecular)
						glUniform4fv(uniforms->uLightSpecularLoc[i], 1, m_parameters.uLightSpecular[i].f);

					m_parameters.uLightMask[i] &= negCurrentProgramMask;
				}
			}//for (int i = 0; i < MAX_LIGHTS; ++i)
		}//if (DoesCurrentProgramUseLighting())
	}


	HQMatrix4* m_viewMatrix;
	HQMatrix4* m_projMatrix;

	HQFixedFunctionParamenters m_parameters;

	GLuint m_vertexShader[8];
	GLuint m_pixelShader[2];
	GLuint m_program[16];
	HQFixedFunctionParametersLocGL * m_uniformLoc[16];//holding uniform locations for each program

	hquint32 m_flags;
	hquint32 m_activeProgramIndex;
};


/*-----------------------------------------------*/



HQFFShaderControllerGL::HQFFShaderControllerGL()
{
	pFFEmu = HQ_NEW HQFixedFunctionShaderGL();

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

	SetFFTransform(HQ_WORLD, &HQMatrix4::IdentityMatrix());
	SetFFTransform(HQ_VIEW, &HQMatrix4::IdentityMatrix());
	SetFFTransform(HQ_PROJECTION, &HQMatrix4::IdentityMatrix());

	ActiveFFEmu();//active fixed function shader
}

HQFFShaderControllerGL::~HQFFShaderControllerGL()
{
	//no need to delete shaders and buffers because they will be deleted by ShaderManager
	delete pFFEmu;
}

HQReturnVal HQFFShaderControllerGL::ActiveFFEmu()
{
	if (!pFFEmu->IsActive())
	{
		pFFEmu->SetActive(true);
	}
	return HQ_OK;
}

HQReturnVal HQFFShaderControllerGL::DeActiveFFEmu()
{
	if (pFFEmu->IsActive())
	{
#if 0
		//not work, for now, need to explicit supply color and texcoords to fixed function shader
		//deactive color and texcoords attributes if needed
		HQVertexStreamManagerGL *streamMan = static_cast<HQVertexStreamManagerGL*> (g_pOGLDev->GetVertexStreamManager());
		if(streamMan->IsVertexAttribActive(COLOR_ATTRIB) == false){
			glDisableVertexAttribArray(COLOR_ATTRIB);
		}

		if(streamMan->IsVertexAttribActive(TEXCOORD0_ATTRIB) == false){
			glDisableVertexAttribArray(TEXCOORD0_ATTRIB);
		}
#endif
		//deactive emulating shader
		pFFEmu->SetActive(false);
	}
	return HQ_OK;
}

bool HQFFShaderControllerGL::IsFFEmuActive()
{
	return pFFEmu->IsActive();
}

HQReturnVal HQFFShaderControllerGL::SetFFRenderState(HQFFRenderState stateType, const void* pValue)
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

HQReturnVal HQFFShaderControllerGL::SetFFTransform(HQFFTransformMatrix type, const HQBaseMatrix4 *pMatrix)
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

void HQFFShaderControllerGL::NotifyFFRender()// notify shader manager that the render device is going to draw something. Shader manager needs to update Fixed Function emulator if needed
{
#if 0
	//not work, for now, need to explicit supply color and texcoords to fixed function shader
	//active and use default color and texcoords
	HQVertexStreamManagerGL *streamMan = static_cast<HQVertexStreamManagerGL*> (g_pOGLDev->GetVertexStreamManager());
	if(streamMan->IsVertexAttribActive(COLOR_ATTRIB) == false){
		glEnableVertexAttribArray(COLOR_ATTRIB);
		glVertexAttrib4fARB(COLOR_ATTRIB, 1, 1, 1, 1);
	}

	if(streamMan->IsVertexAttribActive(TEXCOORD0_ATTRIB) == false){
		glEnableVertexAttribArray(TEXCOORD0_ATTRIB);
		glVertexAttrib2fARB(TEXCOORD0_ATTRIB, 0, 0);
	}
#endif

	pFFEmu->CommitChange();
}

void HQFFShaderControllerGL::ReleaseFFEmu()
{
	pFFEmu->Release();
}

void HQFFShaderControllerGL::RestoreFFEmu()
{
	pFFEmu->Restore();
}
