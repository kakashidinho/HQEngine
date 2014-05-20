#include "../common_shader_code.h"

/*--------vertex shader------------*/
struct VIN {
	float2 pos : VPOSITION;
};

struct VOUT {
	float4 pos : SV_POSITION;
};

VOUT VS(in VIN input)
{
	VOUT output;

	output.pos.xy = input.pos;
	output.pos.zw = float2(0.0, 1.0);
	return output;
}

/*--------pixel shader------------*/
Texture2D<float4> gbuffer_direct_flux : TEXUNIT0;//direct illumination flux
Texture2D<float4> final_interpolated_illum : TEXUNIT1;//final interpolated indirect illumination

float4 PS(in VOUT input) : SV_Target0
{
	float2 pixelPos = input.pos.xy - float2(0.5, 0.5);
	int2 iPixelPos = int2(pixelPos);
	float4 color = gbuffer_direct_flux.Load(uint3(iPixelPos, 0));
	color += final_interpolated_illum.Load(uint3(iPixelPos, 0));

	return color;
}