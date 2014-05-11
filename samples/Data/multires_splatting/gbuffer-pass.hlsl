#include "../common_shader_code.h"

#define NORMALIZE_RESULTS 0

/*---------------vertex shader-------------*/
cbuffer transform : register (b0) {
	float4x3 worldMat;
	float4x4 viewMat;
	float4x4 projMat;
	float4 cameraPos;//camera's world position
};

cbuffer lightView : register (b1) {
	float4x4 lightViewMat;//light camera's view matrix
	float4x4 lightProjMat;//light camera's projection matrix
};


struct vOut{
	float4 oPos : SV_Position;//clip space position
	float3 oPosW : TEXCOORD0;//world space position
	float3 oNormalW: TEXCOORD1;//world space normal
	float4 oPosLight: TEXCOORD2;//projected position in light camera's clip space
};



//vertex shader
vOut VS(in vPNIn input){
	vOut output;
	
	//world space position and projected position
	transformWVP(input.iPos, worldMat, viewMat, projMat, output.oPosW, output.oPos);
	
	//world space normal
	output.oNormalW = worldTransformNormal(input.iNormal, worldMat);
	
	//light camera's clip space position
	output.oPosLight = clipTransformCoords(output.oPosW, lightViewMat, lightProjMat);
	
	return output;
};

/*-------fragment shader------------------*/
#define DEPTH_BIAS 0.02
#define SHADOW_MAP_SIZE 512

cbuffer materials : register (b2) {
	SpecularMaterial material[7];
};


cbuffer materialIndex : register (b3) {
	int materialID;
};

cbuffer lightProperties : register (b4) {

	float3 lightPosition;
	float3 lightDirection;
	float4 lightAmbient;
	float4 lightDiffuse;
	float3 lightFalloff_cosHalfAngle_cosHalfTheta;
	float4 lightSpecular;
};


struct pIn{
	float4 position: SV_Position; //clip space position
	float3 posW : TEXCOORD0;//world space position
	float3 normalW: TEXCOORD1; //world space normal
	float4 posLight: TEXCOORD2; //projected position in light camera's clip space
};

struct pOut{
	float4 direct_flux: SV_Target0;//direct illumination's flux
	float4 posW: SV_Target1;//world space position
	float4 normalW: SV_Target2;//world space normal
	float2 depth_materialID: SV_Target3;//distance to camera and materialID
};

Texture2D<float4> rsm_depthMap: register (t0);//rsm depth texture
uniform SamplerState rsm_depthMap_sampler: TEXUNIT0;//depth texture


//compute shadow factor
float computeShadowFactor(float2 texcoords, float fragmentDepth){
	return computeShadowFactor(rsm_depthMap, rsm_depthMap_sampler, SHADOW_MAP_SIZE, texcoords, fragmentDepth, DEPTH_BIAS);
}

pOut PS (in pIn input){
	pOut output;

	//calculate texcoords in shadow map
	float2 shadowMapUV = input.posLight.xy / input.posLight.w;
	//scale to [0..1]
	shadowMapUV = scaleToTexcoord(shadowMapUV);
	
	//distance to camera
	float3 toCameraVec = cameraPos.xyz - input.posW;

	//fragment depth with respect to light source
	float fragmentLightDepth = length(input.posW - lightPosition.xyz);

	//shadow factor
	float shadowFactor = computeShadowFactor(shadowMapUV, fragmentLightDepth);
	
	//re-normalize normal
	input.normalW = normalize(input.normalW);

	//direct illumination
	float4 direct_flux;
	float3 lightVec = normalize(input.posW - lightPosition);
	float spot = calculateSpotLightFactor(lightVec, lightDirection, lightFalloff_cosHalfAngle_cosHalfTheta);
	
	/*------diffuse----------*/
	float4 diffuse = lightDiffuse * max(dot(-lightVec, input.normalW), 0.0) * material[materialID].diffuse;

	direct_flux = diffuse;
	/*------specular----------*/

	//combine
	if (length(material[materialID].specular.xyz) > 0.0)
	{
		float3 reflectedVec = reflect(lightVec, input.normalW);
		float specFactor = pow(max(dot(reflectedVec, normalize(toCameraVec)), 0.0), material[materialID].specPower);
		float4 specular = lightSpecular * material[materialID].specular * specFactor;
		direct_flux += specular;
	}
	//multiply with spot light's factor and shadow factor
	direct_flux *= spot * shadowFactor;

	//combining with ambient intensity
	direct_flux += lightAmbient * material[materialID].ambient;

	//store result
	output.posW = float4(input.posW, 1.0);
	output.normalW = float4(input.normalW, 0.0);
	output.direct_flux = direct_flux;
	output.depth_materialID.x = length(toCameraVec);//distance to camera
	output.depth_materialID.y = materialID;

	return output;
}