// geometry shader for growing hair


/*
// task 1
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
	vec4 add = vec4(0.5f, 0.0f, -0.5f, 0.f);
	for(int i=0; i< gl_in.length(); i++){
		gl_Position = gl_in[i].gl_Position + add;
		gl_Position = Projection * View * gl_Position;
		
		EmitVertex();
	}
	EndPrimitive();

	//clone a second bunny
	for(int i=0; i< gl_in.length(); i++){
		gl_Position = gl_in[i].gl_Position - add;
		gl_Position = Projection * View * gl_Position;
		EmitVertex();
	}
	EndPrimitive();

}


*//*
//task 02
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
	for(int i=0; i < gl_in.length(); i++){
		
		//draw lines
		vec3 Point = gl_in[i].gl_Position.xyz;
		vec3 Normal = normalize(normal[i].xyz);
		gl_Position = Projection * View * vec4(Point,1);
		EmitVertex();

        vec4 l = vec4(Point,1);
		l = l + 0.005f * vec4(Normal,0);       //used some random number -> result looks acceptable
		gl_Position = Projection * View * l; 
		EmitVertex();

		//gravity
		for(int j = 1; j < OUT_VERTS; j++)
		{
			vec4 diff = 0.01 * vec4(Normal,0) - 0.003 * j * vec4(0,1,0,0); 
			diff = 0.005 * normalize(diff);
			l = l + diff;
			gl_Position = Projection * View * l;
		    EmitVertex();
		}
		
		EndPrimitive();
	}
}
*/
//task 03

// geometry shader for growing hair

#version 150

#define OUT_VERTS 19



layout(triangles) in;

//added
in vec3 normal[];

//changed to line strip
layout(triangle_strip, max_vertices = OUT_VERTS) out;
out vec3 normal_out;

layout(std140) uniform GlobalMatrices
{
	mat4 Projection;
	mat4 View;
};


void main(void)
{
	


	gl_Position = vec4(0);
	for(int i=0; i< gl_in.length(); i++){
		vec3 lookat = gl_in[i].gl_Position.xyz - vec3(View[0].z,View[1].z,View[2].z);
		//draw lines, start vertex

		vec3 Point = gl_in[i].gl_Position.xyz;
		vec3 Normal = normalize(normal[i].xyz);
		normal_out= vec3(1,1,1);

		gl_Position = Projection * View * vec4(Point,1);
		EmitVertex();

		vec3 r = normalize(cross(Normal,-lookat));
		vec4 t = vec4(Point,1);
		t = t + 0.005 * vec4(Normal,0) + 0.01 * vec4(r,0);
		normal_out = vec3(1,1,1);
		gl_Position = Projection * View * t;
		EmitVertex();

        vec4 l = vec4(Point,1);
		l = l + 0.01 * vec4(Normal,0);       //used some random number -> result looks acceptable
		gl_Position = Projection * View * l; 

		normal_out = vec3(1,1,1);
		EmitVertex();

		//gravity
		for(int j = 2; j < OUT_VERTS; j++)
		{
			vec4 diff = 0.01 *vec4(Normal,0) - 0.003 *j*vec4(0,1,0,0); 
			diff = 0.01*normalize(diff);
			r = normalize(cross(diff.xyz,-lookat));
			
			l = l + diff;
			t = l + 0.005 * vec4(Normal,0) + 0.02 / j * vec4(r,0);
		normal_out = vec3(1,1,1) * 1/j;
			gl_Position = Projection * View * t;
			EmitVertex();
		normal_out = vec3(1,1,1) * 1/j;
			gl_Position = Projection * View * l;
		    EmitVertex();
		}
		
		EndPrimitive();
	}
}
