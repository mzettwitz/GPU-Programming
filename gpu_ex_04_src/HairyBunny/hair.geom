// geometry shader for growing hair

#version 150

#define OUT_VERTS 12




layout(triangles) in;

//added
in vec3 normal[];

//changed to line strip
layout(line_strip, max_vertices = OUT_VERTS) out;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};

void main(void)
{
	
	
	gl_Position = vec4(0);
	for(int i=0; i< gl_in.length(); i++){
		
		//draw lines
		vec3 Point = gl_in[i].gl_Position.xyz;
		vec3 Normal = normalize(normal[i].xyz);
		gl_Position = Projection * View * vec4(Point,1);
		EmitVertex();

        vec4 l = vec4(Point,1);
		l = l + 0.01 * vec4(Normal,0);       //used some random number -> result looks acceptable
		gl_Position = Projection * View * l; 
		EmitVertex();

		//gravity
		for(int j = 1; j < OUT_VERTS; j++)
		{
			vec4 diff = 0.01 *vec4(Normal,0) - 0.003 *j*vec4(0,1,0,0); 
			diff = 0.01*normalize(diff);
			l = l + diff;
			gl_Position = Projection * View * l;
		    EmitVertex();
		}
		
		EndPrimitive();
	}
}