cbuffer Parameters : register(b0)
{
	float4 COLOR;
	float DEPTH;
};

struct VS_OUT
{
	float4 position : SV_Position;
	float4 color : COLOR;
};

VS_OUT VS(in float2 position : POSITION)
{
	VS_OUT vout;
	vout.position = float4(position.x, position.y, DEPTH, 1.0f);
	vout.color = COLOR;
	return vout;
}

float4 PS(in VS_OUT pIn) :SV_Target
{
	return pIn.color;
}