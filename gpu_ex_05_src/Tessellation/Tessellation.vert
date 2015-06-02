#version 400

// input from input assembler.
layout(location = 0) in vec3 vs_in_Position;
layout(location = 1) in vec3 vs_in_Normal;
layout(location = 2) in vec2 vs_in_TexCoord;

// output of the vertex shader.
out vec3 vs_out_Position;
out vec3 vs_out_Normal;
out vec2 vs_out_TexCoord;

void main()
{
	// bypass all data from the input assembler.
	vs_out_Position = vs_in_Position;
	vs_out_Normal = vs_in_Normal;
	vs_out_TexCoord = vs_in_TexCoord;
	gl_Position = vec4(vs_out_Position, 1);
}
