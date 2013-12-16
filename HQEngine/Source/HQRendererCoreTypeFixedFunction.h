#ifndef HQ_CORETYPE_FF_H
#define HQ_CORETYPE_FF_H

#include "HQMiscDataType.h"
#include "HQRendererCoreType.h"


typedef struct _HQFFMaterial : public HQColorMaterial {
	///cast to int*
	operator hqint32 *() {return (hqint32*)this;}
} HQFFMaterial;

typedef enum _HQFFLightType
{
	HQ_LIGHT_POINT         = 1,
	HQ_LIGHT_DIRECTIONAL   = 3,
	HQ_LIGHT_FORCE_DWORD   = 0xffffffff 
} HQFFLightType , HQLightType;

typedef struct _HQFFLight {
	HQFFLightType  type;
	HQColor diffuse;
	HQColor specular;
	HQColor ambient;
	HQFloat3 position;
	HQFloat3 direction;
	hqfloat32 attenuation0;
	hqfloat32 attenuation1;
	hqfloat32 attenuation2;
	
	///constructor
	_HQFFLight() :
		type (HQ_LIGHT_DIRECTIONAL), 
		attenuation0(1) ,attenuation1(0) , attenuation2(0) 
	{
		position.Set(0,0,0);
		direction.Set(0, 0, 1);
		diffuse.r = diffuse.g = diffuse.b = 1; diffuse.a = 0; 
		specular.r = specular.g = specular.b = specular.a = 0;
		ambient.r = ambient.g = ambient.b = ambient.a = 0;
	}
	
	///cast to int*
	operator hqint32*() {return (hqint32*)this;}

} HQFFLight , HQBaseLight;

/*-----------those enums is used by shader manager when no custom shader is used----------*/
//fixed function transform matrix 
//this enum is used as first parameter of ShaderManager's SetUniformMatrix() method (HQMatrix4 version),
//3rd paramter must be 1
//those matrices must be reseted after device lost
typedef enum _HQFFTransformMatrix
{
	HQ_WORLD = 256,//world transform matrix
	HQ_VIEW = 2,//view transform matrix
	HQ_PROJECTION = 3,//projection matrix
	HQ_FFTM_FORCE_DWORD = 0xffffffff
} HQFFTransformMatrix;

//render state
//this enum is used as first parameter of ShaderManager's SetUniformInt() method,
//3rd paramter must be 1
//those states must be reseted after device lost
typedef enum _HQFFRenderState
{
	/*----used in SetUniformInt(3 params)----*/
	HQ_LIGHT0 = 0,//first light, 2nd param is HQFFLight
	HQ_LIGHT1 = 1,//2nd light, 2nd param is HQFFLight
	HQ_LIGHT2 = 2,//3rd light, 2nd param is HQFFLight
	HQ_LIGHT3 = 3,//4th light, 2nd param is HQFFLight
	HQ_MATERIAL = 16,// 2nd param is HQFFMaterial
	
	/*----used in SetUniformInt(2 params)----*/
	HQ_LIGHT0_ENABLE = 8,//disable or enable first light, default is disable , 2nd param is HQBool
	HQ_LIGHT1_ENABLE = 9,//disable or enable 2nd light
	HQ_LIGHT2_ENABLE = 10,//disable or enable 3rd light
	HQ_LIGHT3_ENABLE = 11,//disable or enable 4th light
	HQ_TEXTURE_ENABLE = 17,//disable or enable fixed function texturing, default is enable, 2nd param is HQBool
	HQ_TEXTURE_STAGE = 18,//reserved value , active texture stage for changing its state, 2nd param is in [0 , 7]
	HQ_LIGHTING_ENABLE = 137,//enable or disable lighting, default is disable , 2nd param is HQBool , *same as d3d9 enum value*
	HQ_AMBIENT = 139,//global ambient light color, 2nd param is HQColorui (CL_BGRA), default is black , *same as d3d9 enum value*
	HQ_SPECULAR_ENABLE = 29,//enable or disable specular hightlight ,default is disable , 2nd param is HQBool , *same as d3d9 enum value*
	HQ_NORMALIZE_NORMALS = 143,//enable or disable auto normalize vertex normal , default is disable , 2nd param is HQBool , *same as d3d9 enum value*
	HQ_FFRT_FORCE_DWORD = 0xffffffff
} HQFFRenderState;

#endif