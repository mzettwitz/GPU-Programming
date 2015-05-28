// Framework für GLSL-Programme
// Pumping and Glowing Teapot 

#include <GL/glew.h>
#include <stdlib.h>
#include <GL/freeglut.h>
#include <iostream>
#include <string>
#include <fstream>

using namespace std;

// Global variables
GLfloat alpha = 0;

// GLSL related variables
// Blur Shader Program

GLuint fragmentShaderBlur = -1;
GLuint vertexShaderBlur = -1;
GLuint shaderProgramBlur = -1;

//selbst angelegt
GLuint fragmentShaderBlur_hor = -1;
GLuint fragmentShaderBlur_vert = -1;
GLuint shaderProgramBlur_hor = -1;
GLuint shaderProgramBlur_vert = -1;

// Texture Ids and Framebuffer Object Ids
GLuint teapotTextureId = 0;
GLuint depthTextureId = 0;
GLuint teapotFB = 0;

//selbst angelegt
GLuint blurHorizontalTextureId = 0;
GLuint blurHorizontalFB = 0;

// Window size
int width = 512;
int height = 512;

// uniform locations
GLint teapotTextureLocation;
GLint blurHorizontalTextureLocation;

bool useBlur = true;
bool useBlurSep = true;

// Print information about the compiling step
void printShaderInfoLog(GLuint shader)
{
	if (shader == -1)
		return;

	GLint infologLength = 0;
	GLsizei charsWritten = 0;
	char *infoLog;

	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);
	infoLog = (char *)malloc(infologLength);
	glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
	printf("%s\n", infoLog);
	free(infoLog);
}

// Print information about the linking step
void printProgramInfoLog(GLuint program)
{
	if (program == -1)
		return;

	GLint infoLogLength = 0;
	GLsizei charsWritten = 0;
	char *infoLog;

	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
	infoLog = (char *)malloc(infoLogLength);
	glGetProgramInfoLog(program, infoLogLength, &charsWritten, infoLog);
	printf("%s\n", infoLog);
	free(infoLog);
}

// Reads a file and returns the content as a string
string readFile(string fileName)
{
	string fileContent;
	string line;

	ifstream file(fileName.c_str());
	if (file.is_open()) {
		while (!file.eof()){
			getline(file, line);
			line += "\n";
			fileContent += line;
		}
		file.close();
	}
	else
		cout << "ERROR: Unable to open file " << fileName << endl;

	return fileContent;
}

void initGL()
{
	// Initialize camera
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, 1, 0.1, 100);
	glMatrixMode(GL_MODELVIEW);

	// Initialize light source
	GLfloat light_pos[] = { 10, 10, 10, 1 };
	GLfloat light_col[] = { 1, 1, 1, 1 };

	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_col);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_col);

	// Enable lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	// Initialize material
	GLfloat teapot_diffuse[] = { 0.75f, 0.375f, 0.075f, 1 };
	GLfloat teapot_specular[] = { 0.8f, 0.8f, 0.8f, 1 };

	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, teapot_diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, teapot_specular);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 45.2776f);

	// Enable depth buffer
	glEnable(GL_DEPTH_TEST);
}

void initGLSL()
{
	// TODO: Create empty shader object (vertex shader) and assign it to 'vertexShaderPumping'
	// Read vertex shader source 
	// TODO: Attach shader code
	// TODO: Compile shader	
	// TODO: Create empty shader object (fragment shader) and assign it to 'fragmentShaderPumping'

	//Vertex Shader
	vertexShaderBlur = glCreateShader(GL_VERTEX_SHADER); //<<<< ich vermute mal das oben ist ein Tippfehler?
	string shaderSource = readFile("blur.vert");
	const char* sourcePtr = shaderSource.c_str();
	glShaderSource(vertexShaderBlur, 1, &sourcePtr, NULL);
	glCompileShader(vertexShaderBlur);
	printShaderInfoLog(vertexShaderBlur);

	// Fragment Shader
	fragmentShaderBlur = glCreateShader(GL_FRAGMENT_SHADER);
	shaderSource = readFile("blur.frag");
	sourcePtr = shaderSource.c_str();
	glShaderSource(fragmentShaderBlur, 1, &sourcePtr, NULL);
	glCompileShader(fragmentShaderBlur);
	printShaderInfoLog(fragmentShaderBlur);

	//horizontaler Fragment Shader
	fragmentShaderBlur_hor = glCreateShader(GL_FRAGMENT_SHADER);
	shaderSource = readFile("blur_hor.frag");
	sourcePtr = shaderSource.c_str();
	glShaderSource(fragmentShaderBlur_hor, 1, &sourcePtr, NULL);
	glCompileShader(fragmentShaderBlur_hor);
	printShaderInfoLog(fragmentShaderBlur_hor);

	//vertikaler  Fragment Shader
	fragmentShaderBlur_vert = glCreateShader(GL_FRAGMENT_SHADER);
	shaderSource = readFile("blur_vert.frag");
	sourcePtr = shaderSource.c_str();
	glShaderSource(fragmentShaderBlur_vert, 1, &sourcePtr, NULL);
	glCompileShader(fragmentShaderBlur_vert);
	printShaderInfoLog(fragmentShaderBlur_vert);

	//Shader Program
	shaderProgramBlur = glCreateProgram();
	glAttachShader(shaderProgramBlur, fragmentShaderBlur);
	glAttachShader(shaderProgramBlur, vertexShaderBlur);
	glLinkProgram(shaderProgramBlur);
	printProgramInfoLog(shaderProgramBlur);

	//horizontales Shader Programm
	shaderProgramBlur_hor = glCreateProgram();
	glAttachShader(shaderProgramBlur_hor, fragmentShaderBlur_hor);
	glAttachShader(shaderProgramBlur_hor, vertexShaderBlur);
	glLinkProgram(shaderProgramBlur_hor);
	printProgramInfoLog(shaderProgramBlur_hor);


	//vertikales Shader Programm
	shaderProgramBlur_vert = glCreateProgram();
	glAttachShader(shaderProgramBlur_vert, fragmentShaderBlur_vert);
	glAttachShader(shaderProgramBlur_vert, vertexShaderBlur);
	glLinkProgram(shaderProgramBlur_vert);
	printProgramInfoLog(shaderProgramBlur_vert);

	glUseProgram(shaderProgramBlur);


	// Eingabe in diesen Shader ist die Textur, in die die Szene gerendert wird.
	// An dieser Stelle wird die uniform Location für die Textur-Variable im Shader geholt.


	//// Uniform für Blur Shader
	teapotTextureLocation = glGetUniformLocation(shaderProgramBlur, "texture");
	glUniform1i(teapotTextureLocation, 0);                 //Textur 0
	if (teapotTextureLocation == -1)
		cout << "ERROR: No such uniform teapot" << endl;


	//Uniform für horizontal Blur Shader
	glUseProgram(shaderProgramBlur_hor);
	teapotTextureLocation = glGetUniformLocation(shaderProgramBlur_hor, "texture");
	glUniform1i(teapotTextureLocation, 0);                // Textur 0
	if (teapotTextureLocation == -1)
		cout << "ERROR: No such uniform teapot 1" << endl;

	//Uniform für vertikal Blur Shader
	glUseProgram(shaderProgramBlur_vert);
	teapotTextureLocation = glGetUniformLocation(shaderProgramBlur_vert, "texture");
	glUniform1i(teapotTextureLocation, 1);                //Textur 1!
	if (teapotTextureLocation == -1)
		cout << "ERROR: No such uniform teapot 2" << endl;

}


int initFBOTextures()
{
	// Textur (fuer Teapot Bild) anlegen
	glGenTextures(1, &teapotTextureId);
	glBindTexture(GL_TEXTURE_2D, teapotTextureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// Textur für horizontalen Filter
	glGenTextures(1, &blurHorizontalTextureId);
	glBindTexture(GL_TEXTURE_2D, blurHorizontalTextureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// Depth Buffer Textur anlegen 
	glGenTextures(1, &depthTextureId);
	glBindTexture(GL_TEXTURE_2D, depthTextureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Teapot FBO anlegen und Texturen zuweisen
	glGenFramebuffers(1, &teapotFB);
	glBindFramebuffer(GL_FRAMEBUFFER, teapotFB);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, teapotTextureId, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureId, 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, teapotTextureId); // texture 0 is the teapot color buffer

	// FBO anlegen und Textur zuweisen
	glGenFramebuffers(1, &blurHorizontalFB);
	glBindFramebuffer(GL_FRAMEBUFFER, blurHorizontalFB);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, blurHorizontalTextureId, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureId, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, blurHorizontalTextureId); // Textur 1 für horizontalen Filter

	// check framebuffer status
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	switch (status)
	{
	case GL_FRAMEBUFFER_COMPLETE:
		cout << "FBO complete" << endl;
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED:
		cout << "FBO configuration unsupported" << endl;
		return 1;
	default:
		cout << "FBO programmer error" << endl;
		return 1;
	}
	glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

	return 0;
}

void keyboard(unsigned char key, int x, int y)
{
	// set parameters
	switch (key)
	{
	case 'b':
		useBlur = !useBlur;
		break;

	case 'c':
		useBlurSep = !useBlurSep;                //für separaten Filter
		break;
	}
}

// Bildschirmfuellendes Rechteck zeichnen -> Fragment Program wird fuer jedes Pixel aufgerufen
void drawScreenFillingQuad()
{
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0, 0);
		glVertex2f(-1, -1);
		glTexCoord2f(1, 0);
		glVertex2f(1, -1);
		glTexCoord2f(1, 1);
		glVertex2f(1, 1);
		glTexCoord2f(0, 1);
		glVertex2f(-1, 1);
	}
	glEnd();

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
}

void display()
{
	// Pumping Shader anschalten falls aktiviert
	glUseProgram(0);

	//Szene ohne Filter
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	gluLookAt(10, 7, 10, 0, 0, 0, 0, 1, 0);

	glRotatef(alpha, 0, 1, 0);
	glutSolidTeapot(3);


	//Verschachtelter Filter mit b aktivieren und deaktivieren
	if (useBlur){

		//FBO aktivieren
		glBindFramebuffer(GL_FRAMEBUFFER, teapotFB);// Clear window
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//Szene
		glLoadIdentity();
		gluLookAt(10, 7, 10, 0, 0, 0, 0, 1, 0);
		glRotatef(alpha, 0, 1, 0);
		glutSolidTeapot(3);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glUseProgram(shaderProgramBlur);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawScreenFillingQuad();

		glUseProgram(0);
		glutSolidTeapot(3);

	}

	//Separater Filter mit c aktivieren und deaktivieren
	if (useBlurSep){

		//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< vermutlich gibt es hier ein problem...... 
		// irgendwie klappt es anscheinend nicht so wirklich mit dem Übergeben? 
 

		// FBO 1 aktivieren
		glBindFramebuffer(GL_FRAMEBUFFER, teapotFB);           //vielleicht stimmt die Reihenfolge der FBO's auch nicht oder man braucht Teapot garnicht??
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//Szene
		glLoadIdentity();
		gluLookAt(10, 7, 10, 0, 0, 0, 0, 1, 0);
		glRotatef(alpha, 0, 1, 0);
		glutSolidTeapot(10);                                  //was hier angegeben wird ist ihm relativ Latte....


		//FBO 2 aktivieren horizontal Filter
		glBindFramebuffer(GL_FRAMEBUFFER, blurHorizontalFB);
		glUseProgram(shaderProgramBlur_hor);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawScreenFillingQuad();


		//FBO deaktivieren  vertikal Filter
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glUseProgram(shaderProgramBlur_vert);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		drawScreenFillingQuad();

		glUseProgram(0);

		glutSolidTeapot(2);        //nur der Teil wird gezeichnet



	}

	// Increment rotation angle
	alpha += 1;

	// Swap display buffers
	glutSwapBuffers();
}

void timer(int value)
{
	// Call timer() again in 25 milliseconds
	glutTimerFunc(25, timer, 0);

	// Redisplay frame
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	// Initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("Glowing Teapot");

	// Init glew so that the GLSL functionality will be available
	if (glewInit() != GLEW_OK)
		cout << "GLEW init failed!" << endl;

	// OpenGL/GLSL initializations
	initGL();
	initFBOTextures();
	initGLSL();

	// Register callback functions   
	glutKeyboardFunc(keyboard);
	glutDisplayFunc(display);
	glutTimerFunc(25, timer, 0);     // Call timer() in 25 milliseconds

	// Enter main loop
	glutMainLoop();

	return 0;
}
