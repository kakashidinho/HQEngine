#version 430

#ifdef HQEXT_GLSL_SEPARATE_SHADER
out gl_PerVertex{
	vec4 gl_Position;
	float gl_PointSize;
};
#endif

in vec2 iposition VPOSITION;
in vec2 itexcoords VTEXCOORD0;

out vec2 texcoords;

uniform ubuffer0{
	vec2 translation;
};

void main()
{
	gl_Position.xy = iposition + translation;
	gl_Position.zw = vec2(0.0, 1.0);
	texcoords = itexcoords;
}
