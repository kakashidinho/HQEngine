#version 430
#define SIZE_OF_FLOAT 4
#define MAX_UINT 0xffffffff
const uint VERTEX_STRIDE = (4 * SIZE_OF_FLOAT);

vec4 colorRGBA8ToVec4(uint r, uint g, uint b, uint a){
	vec4 color;
	color.x = (r & 0xff) / 255.0f;
	color.y = (g & 0xff) / 255.0f;
	color.z = (b & 0xff) / 255.0f;
	color.w = (a & 0xff) / 255.0f;

	return color;
}


struct DrawIndirect {
	uint vertexCount;
	uint instanceCount;
	uint firstVertex;
	uint firstInstance;
};

struct Vertex {
	vec2 position;
	vec2 texcoords;
};

layout(binding = 0) buffer draw_indirect_buffer_block{
	DrawIndirect draw_indirect_buffer[];
};

layout(binding = 1) buffer vertex_buffer_block{
	Vertex vertices[];
};

layout(binding = 2) uniform writeonly image2D color_texture;

layout(binding = 3) buffer counter_buffer_block{
	uint counter_buffer;
};

layout(local_size_x = 2, local_size_y = 2) in;
void main() {
	ivec3 threadIdx = ivec3(gl_LocalInvocationID);

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		//first mesh is quad
		draw_indirect_buffer[0].vertexCount = 4;//vertex count
		draw_indirect_buffer[0].instanceCount = 1;//instance count
		draw_indirect_buffer[0].firstVertex = 0;//first vertex
		draw_indirect_buffer[0].firstInstance = 0;//vertex instance

		//second mesh is triangle
		draw_indirect_buffer[1].vertexCount = 3;//vertex count
		draw_indirect_buffer[1].instanceCount = 1;//instance count
		draw_indirect_buffer[1].firstVertex = 1;//first vertex
		draw_indirect_buffer[1].firstInstance = 0;//vertex instance

		counter_buffer ++;
	}

	/*
	vertex will be placed in triangle strip
	vertex format {
		vec2 position;
		vec2 texcoords;
	}

		v0                   v1
		(0,0)--------------(0.5,0)------------(1,0)
		|										|
		|										|
		|										|
		(0,1)--------------(0.5,1)------------(1,1)
		v2                    v3
	*/
	ivec2 imageCoords = threadIdx.xy;
	uint vertexIdx = 2 * threadIdx.y + threadIdx.x;
	vec4 color = colorRGBA8ToVec4(
		(vertexIdx * 33 + counter_buffer) % 255,
		255 - (vertexIdx * 33 + counter_buffer) % 255,
		0, 255);

	vertices[vertexIdx].texcoords = vec2(threadIdx.xy);
	vertices[vertexIdx].position.x = float(threadIdx.x) * 0.5;
	vertices[vertexIdx].position.y = float(threadIdx.y);

	imageStore(color_texture, imageCoords, color);
}