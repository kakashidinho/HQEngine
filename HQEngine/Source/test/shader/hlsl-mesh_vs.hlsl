#pragma pack_matrix( row_major )

#ifdef BLENDWEIGHT3
#	pragma message "BLENDWEIGHT3 defined"
#else
#	pragma message "BLENDWEIGHT3 undefined"
#endif

#if TEXTURE_BUFFER
tbuffer transform : register (t11)
{
	float3x4 rotation;
	float4x4 viewProj;
	float3x4 boneMatrices[36];
};
#else
cbuffer transform : register (b10)
{
	float3x4 rotation;
	float4x4 viewProj;
	float3x4 boneMatrices[36];
};
#endif



struct vOut
{
	float4 position : SV_Position;
	float2 texcoord : TEXCOORD0;
	float4 color : COLOR;
	float psize : PSIZE;
};

struct vIn
{
	float3 position : VPOSITION;
#ifdef BLENDWEIGHT3
	float3 blendWeight: VBLENDWEIGHT;
#else
	float blendWeight: VBLENDWEIGHT;
#endif
	uint4 blendIndices: VBLENDINDICES;
	float4 normal : VNORMAL;
	float2 texcoord0  : VTEXCOORD0;
};


float3 mulMatrix3x4(float3x4 mat, float4 v)
{

	float3 re;
	re.x = dot(v , mat[0]);
	re.y = dot(v , mat[1]);
	re.z = dot(v , mat[2]);
	
	return re;
	
	//return mul(v, mat);
}

vOut VS(in vIn input)
{
	vOut output;
	output.color = float4(1,1,1,1);
	output.texcoord = input.texcoord0 ;
	output.position.w = 1.0;
	float4 pos = float4(input.position.xyz , 1.0);
	uint4 blendIndices = input.blendIndices;
	
#ifdef BLENDWEIGHT3
	float4 blendWeight = float4(input.blendWeight, 1.0 - input.blendWeight.x - input.blendWeight.y - input.blendWeight.z);
	
	output.position.xyz = blendWeight.x * mulMatrix3x4(boneMatrices[blendIndices.x] , pos ) +
			blendWeight.y * mulMatrix3x4(boneMatrices[blendIndices.y] , pos ) +
			blendWeight.z * mulMatrix3x4(boneMatrices[blendIndices.z] , pos ) +
			blendWeight.w * mulMatrix3x4(boneMatrices[blendIndices.w] , pos );
#else
	float2 blendWeight = float2(input.blendWeight, 1.0 - input.blendWeight);
	
	output.position.xyz = blendWeight.x * mulMatrix3x4(boneMatrices[blendIndices.x] , pos ) +
			blendWeight.y * mulMatrix3x4(boneMatrices[blendIndices.y] , pos ) ;
#endif
	
	output.position.xyz = mulMatrix3x4(rotation, output.position);
	
	output.position = mul(output.position , viewProj);
	return output;
}