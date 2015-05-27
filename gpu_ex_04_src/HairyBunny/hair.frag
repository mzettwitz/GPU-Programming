// simple fragment shader that outputs transparent white (as hair color)

#version 150

out vec4 fragColor;
in vec3 normal_out;

void main()
{		
	fragColor = vec4(0.75, 0.375, 0.075, 1) * vec4(normal_out,1);
}
