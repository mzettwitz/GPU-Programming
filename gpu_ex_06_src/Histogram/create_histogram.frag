
const float PIC_HEIGHT	= 512.0;
const float PIC_WIDTH	= 512.0;

void main()
{	
	// TODO: Ein Pixel mit der Helligkeit eines einzelnen Grauwerts ausgeben.
	//DONE

	float scale = 255.0;
	float size = 0.6;
	gl_FragColor = vec4(size/scale, size/scale, size/scale, 1.0);
}
