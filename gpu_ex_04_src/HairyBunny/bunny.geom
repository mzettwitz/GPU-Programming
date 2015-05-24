// geometry shader to clone some bunnys..actually pass through

#version 330

layout(triangles) in;
in vec4 diffuse[], ambient[];
in vec3 normal[], lightDir[];

layout(triangle_strip, max_vertices = 6) out;
out vec4 odiffuse, oambient;
out vec3 onormal, olightDir;

void main(void)
{
	gl_Position = vec4(0);
	vec4 add = vec4(1.f,1.f,1.f,1.f);
	for(int i = 0; i < gl_in.length(); i++){
		gl_Position = gl_in[i].gl_Position;		
		odiffuse = diffuse[i];
		oambient = ambient[i];
		onormal = normal[i];
		olightDir = lightDir[i];
		EmitVertex();
	}
	EndPrimitive();


	for(int i = 0; i < gl_in.length(); i++){
		gl_Position = gl_in[i].gl_Position + add;		
		odiffuse = diffuse[i];
		oambient = ambient[i];
		onormal = normal[i];
		olightDir = lightDir[i];
		EmitVertex();
	}
	EndPrimitive();




}