struct VIN {
	float2 position : VPOSITION;
	float2 texcoords : VTEXCOORD0;
};

struct VOUT{
	float4 position : SV_Position;
	float2 texcoords: TEXCOORD0;
};

cbuffer C0: register(b0){
	float2 translation;
};

VOUT VS(in VIN vin){
	VOUT vout;
	vout.position.xy = vin.position + translation;
	vout.position.zw = float2(0.0, 1.0);
	vout.texcoords = vin.texcoords;

	return vout;
}

Texture2D<float4> colorTex : register(t0);
uniform SamplerState colorTexSampler : TEXUNIT0;

float4 PS(in VOUT pin) : SV_Target{
	float4 color = colorTex.Sample(colorTexSampler, pin.texcoords);
	return color;
}