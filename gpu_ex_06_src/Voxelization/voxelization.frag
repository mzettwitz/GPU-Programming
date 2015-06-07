// Extension aktivieren, damit << im Befehlssatz vorliegt.
#extension GL_EXT_gpu_shader4 : enable


// Ausgabevariable
varying out uvec4 result;

void main()
{	
	// TODO: Tiefenwert von [0..1] auf {0..127} abbilden.
	const vec4 bitSh = vec4(256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0);
	const vec4 bitMsk = vec4(0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0);
	vec4 res = fract(gl_FragCoord.y * bitSh);
	res -= res.xxyz * bitMsk;

	result = uvec4(res);
	
	
	
	// Dies ergibt beispielsweise den Wert 42.
	// Erzeugen Sie nun eine bit-Maske, in der das (im Beispiel) 42te Bit (von rechts gezählt) eine 1 ist und alle anderen eine 0.
	// 00000000..000000010000000..00000000
	// |<- 86 Nullen ->| |<- 41 Nullen ->|
	//                  ^
	//                Bit 42
	// Weisen Sie diese bit-Maske der Variable 'result' zu.
}
