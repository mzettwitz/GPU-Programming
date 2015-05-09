// Framework für GLSL-Programme
// Pumping and Glowing Teapot 

#include <GL/glew.h>
#include <stdlib.h> /* Known bug in GLUT, see http://www.lighthouse3d.com/opengl/glut/ (bottom of page) for an explanation */
#include <GL/freeglut.h>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>

using namespace std;

// Global variables
GLfloat alpha = 0;

// GLSL related variables
// Pumping Shader Program
GLuint vertexShaderPumping = -1;	
GLuint fragmentShaderPumping = -1;
GLuint shaderProgramPumping = -1;

// Window size
int width = 512;       
int height = 512;

// uniform locations
float uniformTime = 0.f;
GLint loc;

bool usePumping = true;

// Print information about the compiling step
void printShaderInfoLog(GLuint shader)
{
	if (shader == -1)
		return;
    GLint infologLength = 0;
    GLsizei charsWritten  = 0;
    char *infoLog;

	glGetShaderiv(shader, GL_INFO_LOG_LENGTH,&infologLength);		
	infoLog = (char *)malloc(infologLength);
	glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
	printf("%s\n",infoLog);
	free(infoLog);
}

// Print information about the linking step
void printProgramInfoLog(GLuint program)
{
	if (program == -1)
		return;

	GLint infoLogLength = 0;
	GLsizei charsWritten  = 0;
	char *infoLog;

	glGetProgramiv(program, GL_INFO_LOG_LENGTH,&infoLogLength);
	infoLog = (char *)malloc(infoLogLength);
	glGetProgramInfoLog(program, infoLogLength, &charsWritten, infoLog);
	printf("%s\n",infoLog);
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
			getline (file,line);
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
   GLfloat light_pos[] = {10, 10, 10, 1};
   GLfloat light_col[] = { 1,  1,  1, 1};

   glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
   glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_col);
   glLightfv(GL_LIGHT0, GL_SPECULAR, light_col);

   // Enable lighting
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);

   // Initialize material
   //GLfloat teapot_diffuse[]  = {0.75f, 0.375f, 0.075f, 1};
   GLfloat teapot_diffuse[] = { 0.08f, 0.5f, 0.075f, 1 };    //HULKMODE
   GLfloat teapot_specular[] = {0.8f, 0.8f, 0.8f, 1};

   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, teapot_diffuse);
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, teapot_specular);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 45.2776f);

   // Enable depth buffer
   glEnable(GL_DEPTH_TEST);
}

void initGLSL()
{
	///////VERTEX SHADER
	// DONE: Create empty shader object (vertex shader) and assign it to 'vertexShaderPumping'
	vertexShaderPumping = glCreateShader(GL_VERTEX_SHADER);
	
	// Read vertex shader source 
	string shaderSource = readFile("pumping.vert");
	const char* sourcePtr = shaderSource.c_str();

	// DONE: Attach shader code
	glShaderSource(vertexShaderPumping, 1, &sourcePtr, NULL);
	
	// DONE: Compile shader	
	glCompileShader(vertexShaderPumping);
	
	printShaderInfoLog(vertexShaderPumping);

	///////FRAGMENT SHADER
	// DONE: Create empty shader object (fragment shader) and assign it to 'fragmentShaderPumping'
	fragmentShaderPumping = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source 
	shaderSource = readFile("pumping.frag");
	sourcePtr = shaderSource.c_str();

	// DONE: Attach shader code
	glShaderSource(fragmentShaderPumping, 1, &sourcePtr, NULL);

	// DONE: Compile shader
	glCompileShader(fragmentShaderPumping);

	printShaderInfoLog(fragmentShaderPumping);

	///////SHADER PROGRAMM
	// DONE: Create shader program and assign it to 'shaderProgramPumping'
	shaderProgramPumping = glCreateProgram();

	// DONE: Attach shader vertex shader and fragment shader to program	
	glAttachShader(shaderProgramPumping, vertexShaderPumping);
	glAttachShader(shaderProgramPumping, fragmentShaderPumping);

	// DONE: Link program
	glLinkProgram(shaderProgramPumping);

	printProgramInfoLog(shaderProgramPumping);

	// DONE: Use program.	
	glUseProgram(shaderProgramPumping);

	// DONE: Teilaufgabe 3... Die Uniform Location der Zeit-Variable bestimmen.	
	loc = glGetUniformLocation(shaderProgramPumping, "time");
	assert(loc != -1);
}

void keyboard(unsigned char key, int x, int y)
{
	// set parameters
	switch (key) 
	{                                 
		case 'p':
			usePumping = !usePumping;
			break;
	}
}

void display()
{	
	// Pumping Shader anschalten falls aktiviert
	if (usePumping) {
		glUseProgram( shaderProgramPumping );
		// DONE: Den Zeitparameter (uniform) aktualisieren
		glUniform1f(loc, uniformTime);
		
	}
	else {
		glUseProgram( 0 );
	}

	// Clear window
	glClearColor(1, 1, 1, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();
	gluLookAt(10, 7, 10, 0, 0, 0, 0, 1, 0);

	glRotatef(alpha, 0, 1, 0);
	glutSolidTeapot(3);

	// Increment rotation angle
	alpha += 1;

	// DONE: Inkrementieren des Zeit Parameters.
	//normal: 0.1f
	//HULKMODE: 0.25f 
	uniformTime += 0.25f;

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
   glutCreateWindow("Pumping Teapot");

   // Init glew so that the GLSL functionality will be available
   if(glewInit() != GLEW_OK)
	   cout << "GLEW init failed!" << endl;

	// OpenGL/GLSL initializations
	initGL();
	initGLSL();

	// Register callback functions   
	glutKeyboardFunc(keyboard);
	glutDisplayFunc(display);
	glutTimerFunc(25, timer, 0);     // Call timer() in 25 milliseconds

	// Enter main loop
	glutMainLoop();

	return 0;
}
