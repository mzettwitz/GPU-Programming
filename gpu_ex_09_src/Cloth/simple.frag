
#version 330

in vec2 out_TexCoord;
in vec3 out_Normal;

out vec4 FragColor;

void main()
{	
	vec3 n = out_Normal;
	vec3 l = vec3(0,0,1);
	float NdotL = abs(dot(n,l));
	NdotL = NdotL * 0.8 + pow(NdotL,16);
	vec3 c = vec3(out_TexCoord * max(length(dFdx(out_TexCoord)), length(dFdy(out_TexCoord))) * 70, 0.5);
	c = c * NdotL;
	FragColor = vec4(c,1);
}
