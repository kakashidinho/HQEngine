#pragma pack_matrix( row_major )

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



Texture2D texture0 : register (t0);
SamplerState sampler0 : register (s15);

float4 PS(in float4 position : SV_Position,
	in float2 texcoord  : TEXCOORD0  , 
	in float4 color : COLOR ,
	in float psize : PSIZE) : SV_Target
{
	return texture0.Sample ( sampler0 , texcoord) * color;
}