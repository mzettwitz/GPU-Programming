uniform sampler2D texture;

void main() 
{
	
	float texCoordDelta = 1. / 512.;
	
	int filterWidth = 55;	
	
	vec2 texCoord;

	texCoord.x = gl_TexCoord[0].s;
	texCoord.y = gl_TexCoord[0].t - (float(filterWidth / 2) * texCoordDelta);

	vec3 val = vec3(0); 

    for(int j=0; j< filterWidth; j++) 
	{
		
		val= val + texture2D(texture, texCoord).xyz;  
		texCoord.y= texCoord.y + texCoordDelta;

	}

	//val = 2.0 * val / float(filterWidth);
	val = 2.0 * val / float(filterWidth*filterWidth);    

	gl_FragColor.rgb = val.xyz;

	// Die folgende Zeile dient nur zu Debugzwecken!
	// Wenn das Framebuffer object richtig eingestellt wurde und die Textur an diesen Shader übergeben wurde
	// wird die Textur duch den folgenden Befehl einfach nur angezeigt.
	//gl_FragColor.rgb = texture2D(texture,gl_TexCoord[0].st).xyz;

	
}