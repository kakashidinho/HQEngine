#pragma pack_matrix( row_major )


#if TEXTURE_BUFFER
tbuffer transform : register (t11)
{
	float3x4 rotation;
	float4x4 viewProj;
};
#else
cbuffer transform : register (b11)
{
	float3x4 rotation;
	float4x4 viewProj;
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
	float4 color0 : VCOLOR;
	float psize : VPSIZE;
	float2 texcoord0  : VTEXCOORD6;
};



vOut VS(in vIn input)
{
	vOut output;
	output.color = float4(1,1,1,1);
	output.texcoord = input.texcoord0 ;
	output.position.w = 1.0;
	float4 pos = float4(input.position.xyz , 1.0);
	output.position.x = dot(pos , rotation[0]);
	output.position.y = dot(pos , rotation[1]);
	output.position.z = dot(pos , rotation[2]);
	output.position = mul(output.position , viewProj);
	output.psize = input.psize;
	return output;
}