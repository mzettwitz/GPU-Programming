// simple fragment shader that outputs transparent white (as hair color)

#version 150

out vec4 fragColor;
in vec3 normal_out;
in vec3 normal2_out;
in vec3 shading_normal;
in vec3 smooth_out;

void main()
{		
	vec3 n = normalize(shading_normal);
	float NdotL = abs(dot(n,normalize(normal_out)));
	float N2dotL = abs(dot(normalize(normal2_out),normalize(normal_out)));
	vec4 color = vec4(0.75, 0.375, 0.075,1) * vec4(smooth_out,1) * (NdotL + 1);
	color = color * 2;
	color += vec4(0.75,0.375,0.075,1) * N2dotL * 0.5;
	color = color / 2;
	fragColor = color;
}
