#include <GL/freeglut.h>

int width = 600;
int height = 600;
int button = 2;

void drawQuad(float x, float y, float z,float r, float g, float b,float a)
{
	glBegin(GL_QUADS);
	glColor4f(r, g, b, a);
	glVertex3f(x,y,z);
	glVertex3f(x+1,y,z);
	glVertex3f(x+1,y+1,z);
	glVertex3f(x,y+1,z);
	glEnd();
}


void display(void)
{
	if (button != 2)
	{
		glClearColor(0, 0, 0, 1);
	}
	else
	{
		glClearColor(1, 1, 1, 1);
	}

	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0);

	// *** Farben mit Alpha Kanal setzen
	glEnable(GL_BLEND);
	switch (button)
	{
		case 0:
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			drawQuad(1.f, 1.f, -2.f, 1.f, 0.f, 0.f, 0.7f);
			drawQuad(0.25f, 0.75f, -1.f, 0.f, 1.f, 0.f, 0.7f);
			drawQuad(0.5f, 0.25f, 0.f, 0.f, 0.f, 1.f, 0.7f);
			break;
		case 1:
			glBlendFunc(GL_ONE, GL_ONE);
			drawQuad(1.f, 1.f, -2.f, 1.f, 0.f, 0.f, 1.f);
			drawQuad(0.25f, 0.75f, -1.f, 0.f, 1.f, 0.f, 1.f);
			drawQuad(0.5f, 0.25f, 0.f, 0.f, 0.f, 1.f, 1.f);
			break;
		case 2:
			glBlendFunc(GL_DST_COLOR,GL_ZERO);
			drawQuad(1.f,1.f,-2.f,1.f,1.f,0.f,1.f);
			drawQuad(0.25f,0.75f,-1.f,0.f,1.f,1.f,1.f);
			drawQuad(0.5f,0.25f,0.f,1.f,0.f,1.f,1.f);
		default: 
			break;
	}


	glFlush();
}

void keyboard(unsigned char key, int x, int y)
{

	if (key == 49)
	{
		++button;
		button %= 3;
	}
}


int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("Blending");
	glutKeyboardFunc(keyboard);

	glutDisplayFunc(display);
	
	glDisable(GL_DEPTH_TEST);

	glViewport(0,0,width,height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 2, 0, 2, 0, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// *** Blending Funktion setzen
	glutMainLoop();
	return 0;
}
