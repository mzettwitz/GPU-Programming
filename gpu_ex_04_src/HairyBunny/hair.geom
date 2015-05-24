// geometry shader for growing hair

#version 150

#define OUT_VERTS 6

layout(triangles) in;
layout(triangle_strip, max_vertices = OUT_VERTS) out;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

void main(void)
{
	//translate the bunny
	gl_Position = vec4(0);
	vec4 add = vec4(1.f,1.f,1.f,1.f);
	for(int i=0; i< gl_in.length(); i++){
		gl_Position = gl_in[i].gl_Position + add;
		gl_Position = Projection * View * gl_Position;
		EmitVertex();
	}
	EndPrimitive();

	//clone a second bunny
	for(int i=0; i< gl_in.length(); i++){
		gl_Position = gl_in[i].gl_Position + add*2;
		gl_Position = Projection * View * gl_Position;
		EmitVertex();
	}
	EndPrimitive();


}
