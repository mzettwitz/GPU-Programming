// Extension aktivieren, damit << im Befehlssatz vorliegt.
#extension GL_EXT_gpu_shader4 : enable



// Ausgabevariable
varying out uvec4 result;

void main()
{	
	// TODO: Tiefenwert von [0..1] auf {0..127} abbilden.
	
	int value = int(gl_FragCoord.z * 127.0);
	int pos = int(value / 32);
	int pos2 = int(value % 33); //because of unsigned

	result = uvec4(0,0,0,0);
	result[pos] = uint(2^(pos2));

	/*
	uint value = uint(gl_FragCoord.z * 127.0);
	uint bitmask = uint(1) << uint(2)^(value);
	uint a = bitmask & uint(0xff);
	uint b = (bitmask >> 8) & uint(0xff);
	uint g = (bitmask >> 16) & uint(0xff);
	uint r = (bitmask >> 24) & uint(0xff);
	result = uvec4(a,b,g,r);
	*/

	// Dies ergibt beispielsweise den Wert 42.
	// Erzeugen Sie nun eine bit-Maske, in der das (im Beispiel) 42te Bit (von rechts gezählt) eine 1 ist und alle anderen eine 0.
	// 00000000..000000010000000..00000000
	// |<- 86 Nullen ->| |<- 41 Nullen ->|
	//                  ^
	//                Bit 42
	// Weisen Sie diese bit-Maske der Variable 'result' zu.
}
