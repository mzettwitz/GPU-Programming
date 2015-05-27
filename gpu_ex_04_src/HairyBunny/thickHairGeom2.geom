#define OUT_VERTS 12




layout(line_strip) in;

//added
in vec3 normal[];

//changed to line strip
layout(triangle_strip, max_vertices = OUT_VERTS * 2 - 1) out;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

void main(void)
{
	
	
	vec3 lookat = vec3(View[8],View[9],View[10]);
	for(int i=0; i< gl_in.length() - 1; i++){
		EmitVertex();
		vec3 r = cross(lookat,gl_in[i+1].gl_Position - gl_in[i].gl_Position);
		gl_Position = 1/2 * (gl_in[i+1].gl_Position - gl_in[i].gl_Position) +((i+0.5) / gl_in.length()) * r;
		EmitVertex();	
	}
	EndPrimitive();
}