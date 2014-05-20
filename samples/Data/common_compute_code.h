#ifdef HQEXT_GLSL /*--------GLSL--------------*/

#define main_proc(x, y, z) layout (local_size_x = x, local_size_y = y, local_size_z = z) in; void main()
#define threadIdx gl_LocalInvocationID
#define globalThreadIdx gl_GlobalInvocationID

//declaration
#define decl_texture2d_f(_name, _binding) layout(binding  = _binding) uniform sampler2D _name;
#define decl_texture2d_2f(_name, _binding) layout(binding  = _binding) uniform sampler2D _name;
#define decl_texture2d_4f(_name, _binding) layout(binding  = _binding) uniform sampler2D _name;
#define decl_texture2darray_f(_name, _binding) layout(binding  = _binding) uniform sampler2DArray _name;
#define decl_texture2d_with_sampler_4f(_name, _binding) layout(binding  = _binding) uniform sampler2D _name;
#define decl_texture2d_with_sampler_2f(_name, _binding) layout(binding  = _binding) uniform sampler2D _name;
#define decl_texture2darray_with_sampler_f(_name, _binding) layout(binding  = _binding) uniform sampler2DArray _name;
#define decl_texture2darray_with_sampler_4f(_name, _binding) layout(binding  = _binding) uniform sampler2DArray _name;
#define decl_rwtexture2d_readable_f(_name, _binding) layout(binding  = _binding, r32f) uniform image2D _name;
#define decl_rwtexture2d_readable_rgba8(_name, _binding) layout(binding  = _binding, rgba8) uniform image2D _name;
#define decl_rwtexture2d_readable_4f(_name, _binding) layout(binding  = _binding, rgba32f) uniform image2D _name;
#define decl_rwtexture2d_f(_name, _binding) layout(binding  = _binding) uniform writeonly image2D _name;
#define decl_rwtexture2d_2f(_name, _binding) layout(binding  = _binding) uniform writeonly image2D _name;
#define decl_rwtexture2darray_f(_name, _binding) layout(binding  = _binding) uniform writeonly image2DArray _name;
#define decl_rwtexture2d_4f(_name, _binding) layout(binding  = _binding) uniform writeonly image2D _name;
#define decl_rwtexture2darray_ui(_name, _binding) layout(binding  = _binding) uniform writeonly uimage2DArray _name;
#define decl_rwtexture2d_i(_name, _binding) layout(binding  = _binding) uniform writeonly iimage2D _name;

#define decl_uintbuffer(_name, _binding) layout(binding  = _binding, std430) buffer stbuffer ## _binding {uint _name[];};
#define decl_bytebuffer(_name, _binding) layout(binding  = _binding, std430) buffer stbuffer ## _binding {uint _name[];};

#define decl_structbuffer(_type, _name, _binding) layout(binding  = _binding, std430) buffer stbuffer ## _binding {_type _name[];};

#define begin_decl_uniform_buffer(_name, _binding) layout(binding  = _binding, std140) uniform _name
#define begin_decl_array_with_init(_type, _name, _size) _type _name[_size] = _type[](
#define end_decl_array_with_init );

//texture/image load/store
#define texture2d_read_f(_name, _2dcoords, _lod) (texelFetch(_name, ivec2(_2dcoords), int(_lod)).x) 
#define texture2darray_read_f(_name, _3dcoords, _lod) (texelFetch(_name, ivec3(_3dcoords), int(_lod)).x) 
#define texture2d_read_2f(_name, _2dcoords, _lod) (texelFetch(_name, ivec2(_2dcoords), int(_lod)).xy )
#define texture2d_read_4f(_name, _2dcoords, _lod) texelFetch(_name, ivec2(_2dcoords), int(_lod)) 
#define rwtexture2darray_store_f(_name, _coords, value) imageStore(_name, ivec3(_coords), float4(float(value), 0.0, 0.0, 0.0))
#define rwtexture2darray_store_ui(_name, _coords, value) imageStore(_name, ivec3(_coords), uvec4(uint(value), 0, 0, 0))
#define rwtexture2d_store_i(_name, _coords, value) imageStore(_name, ivec2(_coords), ivec4(int(value), 0, 0, 0))
#define rwtexture2d_store_2f(_name, _coords, value) imageStore(_name, ivec2(_coords), float4(value.xy, 0.0, 0.0))
#define rwtexture2d_store_4f(_name, _coords, value) imageStore(_name, ivec2(_coords), value)
#define rwtexture2d_read_4f(_name, _coords) imageLoad(_name, ivec2(_coords))
#define rwtexture2d_store_rgba8(_name, _coords, value) imageStore(_name, ivec2(_coords), value)
#define rwtexture2d_read_rgba8(_name, _coords) imageLoad(_name, ivec2(_coords))

//texture sampling
#define texture2d_sample_lod_4f(_name, _2dcoords, _lod) (textureLod(_name, vec2(_2dcoords), float(_lod)))
#define texture2darray_sample_lod_f(_name, _3dcoords, _lod) (textureLod(_name, vec3(_3dcoords), float(_lod)).x)
#define texture2darray_sample_lod_4f(_name, _3dcoords, _lod) (textureLod(_name, vec3(_3dcoords), float(_lod)))
#define texture2d_sample_lod_2f(_name, _2dcoords, _lod) (textureLod(_name, vec2(_2dcoords), float(_lod)).xy)

//query size
#define texture2d_getsize_4f(_name, width, height) {ivec2 size = textureSize(_name, 0); width = size.x; height = size.y;}
#define texture2d_getsize_2f(_name, width, height) {ivec2 size = textureSize(_name, 0); width = size.x; height = size.y;}
#define rwtexture2d_getsize_4f(_name, width, height) {ivec2 size = imageSize(_name); width = size.x; height = size.y;}

//byte buffer store
#define bytebuffer_store(buffer, idx, value) {buffer[idx] = (value);}
#define bytebuffer_store2(buffer, idx, value) {buffer[idx] = value.x; buffer[idx+1] = value.y;}
#define bytebuffer_store3(buffer, idx, value) {buffer[idx] = value.x; buffer[idx+1] = value.y; buffer[idx+2] = value.z;}
#define bytebuffer_store4(buffer, idx, value) {buffer[idx] = value.x; buffer[idx+1] = value.y; buffer[idx+2] = value.z; buffer[idx+3] = value.w}


#define bytebuffer_read(buffer, idx) (buffer[idx])
#define bytebuffer_read2(buffer, idx) (uvec2(buffer[idx], buffer[idx+1]))
#define bytebuffer_read3(buffer, idx) (uvec3(buffer[idx], buffer[idx+1], buffer[idx+2]))
#define bytebuffer_read4(buffer, idx) (uvec4(buffer[idx], buffer[idx+1], buffer[idx+2], buffer[idx+3]))

#define asuint floatBitsToUint
#define asfloat uintBitsToFloat

//atomic operations
#define atomic_add(data_mem, added_value, original_value_out) {original_value_out = atomicAdd(data_mem, added_value);}
#define atomic_max(data_mem, value, original_value_out) {original_value_out = atomicMax(data_mem, value);}
#define atomic_min(data_mem, value, original_value_out) {original_value_out = atomicMin(data_mem, value);}
#define atomic_comp_swap(data_mem, compare, value, original_value_out) {original_value_out = atomicCompSwap(data_mem, compare, value);}

//mapping between hlsl and glsl
#define float4 vec4
#define float3 vec3
#define float2 vec2
#define int4   ivec4
#define int3   ivec3
#define int2   ivec2
#define uint4   uvec4
#define uint3   uvec3
#define uint2   uvec2

#define float4x4 mat4
#define float3x4 mat4x3
#define float4x3 mat3x4

#define _LOOP_ATTRIB_
#define _UAV_COND_ATTRIB_
#define _GLOBAL_CONST_ const

//store float4 to one component float texture being viewed as 4 components one
void rwtexture2d_f_store_4f(layout(r32f) image2D _name, uint2 _coords, float4 value)
{
	int2 scaledCoords = int2(4 * _coords.x, _coords.y);
	imageStore(_name, scaledCoords, value.xxxx);
	imageStore(_name, scaledCoords + int2(1, 0), value.yyyy);
	imageStore(_name, scaledCoords + int2(2, 0), value.zzzz);
	imageStore(_name, scaledCoords + int2(3, 0), value.wwww);
}

//read float4 from one component float texture being viewed as 4 components one
float4 rwtexture2d_f_read_4f(layout(r32f) image2D _name, uint2 _coords){
	float4 result;
	int2 scaledCoords = int2(4 * _coords.x, _coords.y);

	result.x = imageLoad(_name, scaledCoords).x;
	result.y = imageLoad(_name, scaledCoords + int2(1, 0)).x;
	result.z = imageLoad(_name, scaledCoords + int2(2, 0)).x;
	result.w = imageLoad(_name, scaledCoords + int2(3, 0)).x;

	return result;
}

//get size of one component texture when being viewed as 4 components one 
void rwtexture2d_f_getsize_4f(layout (r32f) image2D _name, out uint width, out uint height) {
	ivec2 size = imageSize(_name);
	width = size.x / 4;
	height = size.y;
}

#else           /*--------HLSL--------------*/
#define main_proc(x, y, z) [numthreads(x, y, z)] void main(uint3 threadIdx : SV_GroupThreadID, uint3 globalThreadIdx: SV_DispatchThreadID)

//declaration
#define decl_texture2d_f(_name, _binding) Texture2D<float> _name : register(t ## _binding);
#define decl_texture2d_2f(_name, _binding) Texture2D<float2> _name : register(t ## _binding);
#define decl_texture2d_4f(_name, _binding) Texture2D<float4> _name : register(t ## _binding);
#define decl_texture2d_with_sampler(_type, _name, _binding) Texture2D<_type> _name : register(t ## _binding);\
	SamplerState _name ## _sampler_state : TEXUNIT ## _binding;

#define decl_texture2darray_with_sampler(_type, _name, _binding) Texture2DArray<_type> _name : register(t ## _binding);\
	SamplerState _name ## _sampler_state : TEXUNIT ## _binding;

#define decl_texture2d_with_sampler_4f(_name, _binding) decl_texture2d_with_sampler(float4, _name, _binding)
#define decl_texture2d_with_sampler_2f(_name, _binding) decl_texture2d_with_sampler(float2, _name, _binding)
#define decl_texture2darray_with_sampler_f(_name, _binding) decl_texture2darray_with_sampler(float, _name, _binding)
#define decl_texture2darray_with_sampler_4f(_name, _binding) decl_texture2darray_with_sampler(float4, _name, _binding)

#define decl_rwtexture2d_readable_f(_name, _binding) RWTexture2D<float> _name : register(u ## _binding);
#define decl_rwtexture2d_readable_rgba8(_name, _binding) RWTexture2D<uint> _name : register(u ## _binding);
#define decl_rwtexture2d_readable_4f(_name, _binding) error_float4_is_not_readable
#define decl_rwtexture2d_f(_name, _binding) RWTexture2D<float> _name : register(u ## _binding);
#define decl_rwtexture2darray_f(_name, _binding) RWTexture2DArray<float> _name : register(u ## _binding);
#define decl_rwtexture2darray_ui(_name, _binding) RWTexture2DArray<uint> _name : register(u ## _binding);
#define decl_rwtexture2d_i(_name, _binding) RWTexture2D<int> _name : register(u ## _binding);
#define decl_rwtexture2d_2f(_name, _binding) RWTexture2D<float2> _name : register(u ## _binding);
#define decl_rwtexture2d_4f(_name, _binding) RWTexture2D<float4> _name : register(u ## _binding);

#define decl_uintbuffer(_name, _binding) RWBuffer<uint> _name : register(u ## _binding);
#define decl_bytebuffer(_name, _binding) RWByteAddressBuffer _name : register(u ## _binding);
#define decl_structbuffer(_type, _name, _binding) RWStructuredBuffer<_type> _name : register(u ## _binding);

#define begin_decl_uniform_buffer(_name, _binding) cbuffer _name : register(b ## _binding)
#define begin_decl_array_with_init(_type, _name, _size) _type _name[_size] = {
#define end_decl_array_with_init };

//texture/image load/store
#define texture2d_read_f(_name, _2dcoords, _lod) _name.Load(int3(_2dcoords, _lod))
#define texture2d_read_2f(_name, _2dcoords, _lod) _name.Load(int3(_2dcoords, _lod))
#define texture2d_read_4f(_name, _2dcoords, _lod) _name.Load(int3(_2dcoords, _lod))
//texture sampling
#define texture2d_sample_lod_4f(_name, _2dcoords, _lod) _name.SampleLevel(_name ## _sampler_state, _2dcoords, _lod)
#define texture2darray_sample_lod_f(_name, _3dcoords, _lod) _name.SampleLevel(_name ## _sampler_state, _3dcoords, _lod)
#define texture2darray_sample_lod_4f(_name, _3dcoords, _lod) _name.SampleLevel(_name ## _sampler_state, _3dcoords, _lod)
#define texture2d_sample_lod_2f(_name, _2dcoords, _lod) texture2d_sample_lod_4f(_name, _2dcoords, _lod)

//store float4 to one component float texture being viewed as 4 components one
void rwtexture2d_f_store_4f(RWTexture2D<float> _name, uint2 _coords, float4 value) 
{ 
	uint2 scaledCoords = uint2(4 * _coords.x, _coords.y);
	_name[scaledCoords] = value.x;
	_name[scaledCoords + uint2(1, 0)] = value.y;
	_name[scaledCoords + uint2(2, 0)] = value.z;
	_name[scaledCoords + uint2(3, 0)] = value.w;
}

void rwtexture2d_store_2f(RWTexture2D<float2> _name, uint2 _coords, float2 value)
{
	_name[_coords] = value;
}

void rwtexture2d_store_4f(RWTexture2D<float4> _name, uint2 _coords, float4 value)
{
	_name[_coords] = value;
}

void rwtexture2d_store_i(RWTexture2D<int> _name, uint2 _coords, int value)
{
	_name[_coords] = value;
}

void rwtexture2darray_store_f(RWTexture2DArray<float> _name, uint3 _coords, float value)
{
	_name[_coords] = value;
}

void rwtexture2darray_store_ui(RWTexture2DArray<uint> _name, uint3 _coords, uint value)
{
	_name[_coords] = value;
}

//read float4 from one component float texture being viewed as 4 components one
float4 rwtexture2d_f_read_4f(RWTexture2D<float> _name, uint2 _coords){
	float4 result;
	uint2 scaledCoords = uint2(4 * _coords.x, _coords.y);

	result.x = _name[scaledCoords];
	result.y = _name[scaledCoords + uint2(1, 0)];
	result.z = _name[scaledCoords + uint2(2, 0)];
	result.w = _name[scaledCoords + uint2(3, 0)];

	return result;
}

void rwtexture2d_store_rgba8(RWTexture2D<uint> _name, uint2 _coords, float4 color4f)
{
	uint color;
	color4f = min(color4f, float4(1.0, 1.0, 1.0, 1.0));

	color = uint(color4f.x * 255) & 0xff;
	color |= (uint(color4f.y * 255) & 0xff) << 8;
	color |= (uint(color4f.z * 255) & 0xff) << 16;
	color |= (uint(color4f.w * 255) & 0xff) << 24;

	_name[_coords] = color;
}

float4 rwtexture2d_read_rgba8(RWTexture2D<uint> _name, uint2 _coords)
{
	float4 color;
	uint colorrgba8 = _name[_coords];

	color.x = (colorrgba8 & 0xff) / 255.0f;
	color.y = ((colorrgba8 >> 8) & 0xff) / 255.0f;
	color.z = ((colorrgba8 >> 16) & 0xff) / 255.0f;
	color.w = ((colorrgba8 >> 24) & 0xff) / 255.0f;

	return color;
}

//get size of one component texture when being viewed as 4 components one 
void rwtexture2d_f_getsize_4f(RWTexture2D<float> _name, out uint width, out uint height) {
	_name.GetDimensions(width, height);
	width /= 4;
}

void rwtexture2d_getsize_4f(RWTexture2D<float4> _name, out uint width, out uint height) {
	_name.GetDimensions(width, height);
}

void texture2d_getsize_4f(const Texture2D<float4> _name, out uint width, out uint height) {
	_name.GetDimensions(width, height);
}

void texture2d_getsize_2f(const Texture2D<float2> _name, out uint width, out uint height) {
	_name.GetDimensions(width, height);
}


//byte buffer store
#define bytebuffer_store(buffer, idx, value) buffer.Store(4 * idx, (value))
#define bytebuffer_store2(buffer, idx, value) buffer.Store2(4 * idx, (value))
#define bytebuffer_store3(buffer, idx, value) buffer.Store3(4 * idx, (value))
#define bytebuffer_store4(buffer, idx, value) buffer.Store4(4 * idx, (value))


#define bytebuffer_read(buffer, idx) (buffer.Load(4 * idx))
#define bytebuffer_read2(buffer, idx) (buffer.Load2(4 * idx))
#define bytebuffer_read3(buffer, idx) (buffer.Load3(4 * idx))
#define bytebuffer_read4(buffer, idx) (buffer.Load4(4 * idx))

//atomic operations
#define atomic_add(data_mem, added_value, original_value_out) InterlockedAdd(data_mem, added_value, original_value_out)
#define atomic_max(data_mem, value, original_value_out) InterlockedMax(data_mem, value, original_value_out)
#define atomic_min(data_mem, value, original_value_out) InterlockedMin(data_mem, value, original_value_out)
#define atomic_comp_swap(data_mem, compare, value, original_value_out) InterlockedCompareExchange(data_mem, compare, value, original_value_out)

#define _LOOP_ATTRIB_ [loop]
#define _UAV_COND_ATTRIB_ [allow_uav_condition]
#define _GLOBAL_CONST_ static const

#endif