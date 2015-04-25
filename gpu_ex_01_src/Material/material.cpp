// *** Materialien und Lichter

#include <math.h>
#include <GL/freeglut.h>

#define PI 3.141592f

#define ROTATE 1
#define MOVE 2

int width = 600;
int height = 600;

float theta = PI / 2.0f - 0.01f;
float phi = 0.0f;
float distance = 2.5f;
float oldX, oldY;
int motionState;

// Enumeration von Farben.
enum EColor
{
	Red,	// = 0
	Green,	// = 1
	Blue,	// = 2
	Orange, // = 3
	Teal,	// = 4
	Purple	// = 5
};

// Arrays mit Materialeigenschaften.
GLfloat diffuse[6][4] = { {1.0f, 0.2f, 0.2f, 1.0f}, {0.2f, 1.0f, 0.2f, 1.0f}, {0.2f, 0.2f, 1.0f, 1.0f}, {1.0f, 0.8f, 0.2f, 1.0f}, {0.2f, 1.0f, 0.8f, 1.0f}, {0.8f, 0.2f, 1.0f, 1.0f} };
GLfloat specular[6][4] = { {1.0f, 0.7f, 0.7f, 1.0f}, {0.7f, 1.0f, 0.7f, 1.0f}, {0.7f, 0.7f, 1.0f, 1.0f}, {1.0f, 0.8f, 0.7f, 1.0f}, {0.7f, 1.0f, 0.8f, 1.0f}, {0.8f, 0.7f, 1.0f, 1.0f} };
GLfloat ambient[4] = {0.1f, 0.1f, 0.1f, 1.0f};
GLfloat shininess[6] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f};

// Arrays mit standard Lichteigenschaften
GLfloat l_ambient[] = {0.0f, 0.0f, 0.0f, 1.0f}; 
GLfloat l_diffuse[] = {1.0f, 1.0f, 1.0f, 1.0f};
GLfloat l_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};

void applyMaterial(const EColor& colorID)
{
	// Die Material-Eigenschaften anwenden (diffuse, specular, shininess und ambient).
	glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse[colorID]);
	glMaterialfv(GL_FRONT, GL_SPECULAR, specular[colorID]);
	glMaterialfv(GL_FRONT, GL_AMBIENT, ambient);
	glMaterialfv(GL_FRONT, GL_SHININESS, &shininess[colorID]);
}

// Untergrund zeichen: Ein Viereck
void drawGround()
{
	float size = 5;
	int resolution = 20;
	glBegin(GL_QUADS);
	glNormal3f(0,1,0);
	for (int x=-resolution; x<resolution; ++x)
	{
		for (int y=-resolution; y<resolution; ++y)
		{	
			glVertex3f((x+1) / (float)resolution * size, 0, (y+1) / (float)resolution * size);
			glVertex3f((x+1) / (float)resolution * size, 0,  y    / (float)resolution * size);
			glVertex3f( x    / (float)resolution * size, 0,  y    / (float)resolution * size);
			glVertex3f( x    / (float)resolution * size, 0, (y+1) / (float)resolution * size);
		}
	}
	glEnd();
}

void display(void)
{
	// Buffer clearen
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// View Matrix erstellen
	glLoadIdentity();
	float x = distance * sin(theta) * cos(phi);
	float y = distance * cos(theta);
	float z = distance * sin(theta) * sin(phi);
	gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);


	// TODO: Platzieren eines Punktlichtes, das sich auf einer Kreisbahn mit Radius 2.5f in der Höhe 0.5 um (0,0,0) dreht.
	// Die Intensität soll quadratisch mit dem Faktor 0.6 abnehmen. (Hinweis: Nutzen Sie dazu die Quadratic Attentuation)
	GLfloat l1_position[] = { 2.5f, 0.0f, 0.5f, 1.0f };  // <<<<<fixed position for test purposes

	glLightfv(GL_LIGHT1, GL_AMBIENT, l_ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, l_diffuse);
	glLightfv(GL_LIGHT1, GL_SPECULAR, l_specular);
	glLightfv(GL_LIGHT1, GL_POSITION, l1_position);
	glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 0.6f);

	// TODO: Plazieren eines Richtungslichtes aus der Richtung (1,1,1) mit einer Diffusfarbe von (0.2f, 0.2f, 0.2f)
	GLfloat l2_diffuse[] = { 0.2f, 0.2f, 0.2f, 1.f };
	GLfloat l2_position[] = { 1.0f, 1.0f, 1.0f, 0.0f };

	glLightfv(GL_LIGHT2, GL_AMBIENT, l_ambient);
	glLightfv(GL_LIGHT2, GL_DIFFUSE, l2_diffuse);
	glLightfv(GL_LIGHT2, GL_SPECULAR, l_specular);
	glLightfv(GL_LIGHT2, GL_POSITION, l2_position);

	// TODO: Richten eines Spot-Lichtes von (2,2,2) nach (0,0,0) mit einem Cutoff-Winkel von 45°, einem Spot-Exponent von 16 und mit der
	// Diffusfarbe 'Orange'.
	GLfloat l3_position[] = { 2.0f, 2.0f, 2.0f, 0.0f };
	GLfloat l3_direction[] = { 0.f, 0.f, 0.f };

	glLightfv(GL_LIGHT3, GL_AMBIENT, l_ambient);
	glLightfv(GL_LIGHT3, GL_DIFFUSE, diffuse[Orange]);
	glLightfv(GL_LIGHT3, GL_SPECULAR, l_specular);
	glLightfv(GL_LIGHT3, GL_POSITION, l2_position);
	glLightfv(GL_LIGHT3, GL_SPOT_DIRECTION, l3_direction);
	glLightf(GL_LIGHT3, GL_SPOT_CUTOFF, 45.f);
	glLightf(GL_LIGHT3, GL_SPOT_EXPONENT, 16.f);
	
	// Aquamarin (Teal)-farbenen Boden rendern.
	applyMaterial(Teal);
	drawGround();

	// Orange Teekanne rendern bei Position (0,0.5f,0) mit Größe 0.5f
	applyMaterial(Orange);
	glPushMatrix();
	glTranslatef(0, 0.5f, 0);
	glutSolidTeapot(0.5f);
	glPopMatrix();

	// TODO: Rote Box rendern bei Position(3.0f, 0.5f, 3.0f) mit Größe 1.0f
	
	// TODO: Grünen Torus rendern bei Position(-3.0f, 0.5f, 3.0f) mit innerem Radius 0.15f, äußerem Radius 0.35f und jeweils 32 Seiten und Ringen
	
	// TODO: Violette (Purple) Kugel rendern bei Position(-3.0f, 0.5f, -3.0f) mit Radius 0.5f und jeweils 32 Längen- und Breitensegmenten
	
	// TODO: Blauen AUFRECHTEN Kegel rendern bei Position(3.0f, 0.0f, -3.0f) mit Basisradius 0.5, Höhe 1, 32 Ringsegmenten und 4 Höhensegmenten
	
	glutSwapBuffers();	
}

void mouseMotion(int x, int y)
{
	float deltaX = x - oldX;
	float deltaY = y - oldY;
	
	if (motionState == ROTATE) {
		theta -= 0.01f * deltaY;

		if (theta < 0.01f) theta = 0.01f;
		else if (theta > PI/2.0 - 0.01f) theta = PI/2.0f - 0.01f;

		phi += 0.01f * deltaX;	
		if (phi < 0) phi += 2*PI;
		else if (phi > 2*PI) phi -= 2*PI;
	}
	else if (motionState == MOVE) {
		distance += 0.01f * deltaY;
	}

	oldX = (float)x;
	oldY = (float)y;

	glutPostRedisplay();

}

void mouse(int button, int state, int x, int y)
{
	oldX = (float)x;
	oldY = (float)y;

	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			motionState = ROTATE;
		}
	}
	else if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) {
			motionState = MOVE;
		}
	}
}

void idle(void)
{
	glutPostRedisplay();
}


int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("Materialien und Lichter");

	glutDisplayFunc(display);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutIdleFunc(idle);

	// Aktivieren der zusätzlichen Lichter.
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHT2);
	glEnable(GL_LIGHT3);
	
	
	glEnable(GL_DEPTH_TEST);

	glViewport(0,0,width,height);					
	glMatrixMode(GL_PROJECTION);					
	glLoadIdentity();								

	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutMainLoop();
	return 0;
}
