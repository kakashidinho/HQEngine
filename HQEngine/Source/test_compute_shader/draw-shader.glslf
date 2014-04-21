#version 430
in vec2 texcoords;
out vec4 color;

uniform sampler2D colorTex TEXUNIT0;

void main(){
	color = textureLod(colorTex, texcoords, 0.0);
}