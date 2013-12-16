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
	float4 color0 : VCOLOR;
	float psize : VPSIZE;
	float2 texcoord0  : VTEXCOORD6;
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