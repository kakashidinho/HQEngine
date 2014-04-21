#define SIZE_OF_FLOAT 4
#define MAX_UINT 0xffffffff
static const uint VERTEX_STRIDE = (4 * SIZE_OF_FLOAT);

uint colorRGBA(float r, float g, float b, float a){
	uint color;
	color = uint(r * 255) & 0xff;
	color |= (uint(g * 255) & 0xff) << 8;
	color |= (uint(b * 255) & 0xff) << 16;
	color |= (uint(a * 255) & 0xff) << 24;

	return color;
}

uint colorRGBA8(uint r, uint g, uint b, uint a){
	uint color;
	color = r & 0xff;
	color |= (g & 0xff) << 8;
	color |= (b & 0xff) << 16;
	color |= (a & 0xff) << 24;

	return color;
}

RWBuffer<uint> draw_indirect_buffer : register(u0);
RWByteAddressBuffer vertex_buffer : register(u1);
RWTexture2D<uint> color_texture : register(u2);
RWStructuredBuffer<uint> counter_buffer: register (u3);

[numthreads(2, 2, 1)]
void CS(uint3 threadIdx : SV_GroupThreadID) {
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		//first mesh is quad
		draw_indirect_buffer[0] = 4;//vertex count
		draw_indirect_buffer[1] = 1;//instance count
		draw_indirect_buffer[2] = 0;//first vertex
		draw_indirect_buffer[3] = 0;//vertex instance

		//second mesh is triangle
		draw_indirect_buffer[4] = 3;//vertex count
		draw_indirect_buffer[5] = 1;//instance count
		draw_indirect_buffer[6] = 1;//first vertex
		draw_indirect_buffer[7] = 0;//vertex instance

		counter_buffer[0] ++;
	}

	/*
	vertex will be placed in triangle strip
	vertex format {
		float2 position;
		float2 texcoords;
	}

		v0                   v1
		(0,0)--------------(0.5,0)------------(1,0)
		|										|
		|										|
		|										|
		(0,1)--------------(0.5,1)------------(1,1)
		v2                    v3
	*/
	uint2 imageCoords = threadIdx.xy;
	uint vertexIdx = 2 * threadIdx.y + threadIdx.x;
	float2 texcoords = float2(threadIdx.xy);
	uint color = colorRGBA8(
	(vertexIdx * 33 + counter_buffer[0]) % 255,
	255 - (vertexIdx * 33 + counter_buffer[0]) % 255,
		0, 255);
	float2 position;

	position.x = float(threadIdx.x) * 0.5;
	position.y = float(threadIdx.y);

	vertex_buffer.Store2(VERTEX_STRIDE * vertexIdx, asuint(position));
	vertex_buffer.Store2(VERTEX_STRIDE * vertexIdx + 2 * SIZE_OF_FLOAT, asuint(texcoords));

	color_texture[imageCoords] = color;
}