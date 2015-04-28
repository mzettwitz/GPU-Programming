﻿// *** Transformationen

#include <math.h>
#include <GL/freeglut.h>

#define PI 3.141592f

#define ROTATE 1
#define MOVE 2

int width = 600;
int height = 600;

float theta = PI / 2.0f - 0.4f;
float phi = 0.0f;
float distance = 25.0f;
float oldX, oldY;
int motionState;

// Winkel, der sich kontinuierlich erhöht. (Kann für die Bewegungen auf den Kreisbahnen genutzt werden)
float angle = 0.0f;
float speed = 1.f;

float toDeg(float angle) { return angle / PI * 180.0f; }
float toRad(float angle) { return angle * PI / 180.0f; }

// Zeichnet einen Kreis mit einem bestimmten Radius und einer gewissen Anzahl von Liniensegmenten (resolution) in der xz-Ebene.
void drawCircle(float radius, int resolution)
{
	// Abschalten der Beleuchtung.
	glDisable(GL_LIGHTING);

	glBegin(GL_LINE_STRIP);
	for (int i = 0; i <= resolution; ++i)
	{
		float alpha = float(i * (360.f / float(resolution)));
		glVertex3f(radius *cos(toRad(alpha)), 0, radius *sin(toRad(alpha)));
		glColor3f(1.f, 0.f, 0.f);
	}
	glEnd();

	// Anschalten der Beleuchtung.
	glEnable(GL_LIGHTING);
}

/*
void drawLine(glVertex3f start, glVertex3f end)
{
	glBegin(GL_LINE);
	start;
	glColor3f(1.0f, 0.0f, 0.0f, 1.0f);
	end;
	glColor3f(1.0f, 0.0f, 0.0f, 1.0f);
	glEnd();
}
*/

void display(void)
{
	// Buffer clearen
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// View Matrix erstellen
	glLoadIdentity();
	float x = distance * sin(theta) * cos(phi);
	float y = distance * cos(theta);
	float z = distance * sin(theta) * sin(phi);
	gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	// Teekanne rendern.
	glutSolidTeapot(1);

	// Den Matrix-Stack sichern.	
	glPushMatrix();

	// Zeichnen der Kugelkreisbahn.
	drawCircle(10.f, 100);

	// Zeichnen der Kugel.
	// Wenden Sie eine Translation und eine Rotation an, bevor sie die Kugel zeichnen. Sie k�nnen die Variable 'angle' für die Rotation verwenden.
	// Bedenken Sie dabei die richtige Reihenfolge der beiden Transformationen.
	glRotatef(angle * speed, 0, 1, 0);
	glTranslatef(10.f, 0.f, 0.f);
	glutSolidSphere(1, 25, 25);

	// Zeichnen der Würfelkreisbahn.
	// Hinweis: Der Ursprung des Koordinatensystems befindet sich nun im Zentrum des Würfels.
	// Drehen Sie das Koordinatensystem um 90° entlang der Achse, die für die Verschiebung des Würfels genutzt wurde.
	// Danach steht die Würfelkreisbahn senkrecht zur Tangentialrichtung der Kugelkreisbahn.
	glPushMatrix();


	glRotatef(90.f, 1.f, 0.f, 0.f);
	drawCircle(5.f, 100);

	// Zeichnen des Würfels.
	// Wenden Sie die entsprechende Translation und Rotation an, bevor sie den Würfel zeichnen.
	glRotatef(angle*speed, 0, 1, 0);
	glTranslatef(5.f, 0.f, 0.f);

	glutSolidCube(1);


	// Zeichnen einer Linie von Würfel zu Kegel.
	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	glVertex3f(0.f, 0.f, 0.f);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(3.f, 0.f, 0.f);
	glColor3f(1.f, 0.f, 0.f);
	glEnd();
	glEnable(GL_LIGHTING);

	// Zeichnen des Kegels.
	

	// Drehung anwenden, sodass Koordinatensystem in Richtung Ursprung orientiert ist. (Hinweis: Implementieren Sie dies zuletzt.)
	glTranslatef(3.f, 0.f, 0.f);
	glRotatef(90 + angle*speed, 0, -1, 0);

	GLfloat length = 10 + 8 * cosf(angle*speed*PI / 180);
	GLfloat height = 8 * sinf(angle*speed*PI / 180);
	GLfloat heightangle = 180 / PI*atan(height / length);
	glRotatef(heightangle, 0, 1, 0);


	glutSolidCone(0.5, 1, 25, 25);


	// TODO: Zeichnen der Linie von Kegel zu Urpsrung.
	/*glBegin(GL_LINE_STRIP);
	glVertex3f(0, 0, 0);
	glVertex3f(cone_cent_dis * cos(cone_cent_angle), cone_cent_dis * sin(cone_cent_angle), 0);
	glEnd();
	*/
	

	/* MICHI
	//Draw line 
	GLfloat modelview[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
	glVertex3f startLine = modelview * glVertex3f(0, 0, 0);
	glTranslatef(3.f, 0.f, 0.f);
	glVertex3f endLine = modelview * glVertex3f(0, 0, 0);
	drawLine(start, end);
	glTranslatef(3.f, 0.f, 0.f);
	
	
	glPopMatrix();
	glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
	glPushMatrix();
	//should be 0,0,0 anyway
	glVertex3f coneUp = startLine;
	startLine = modelview * glVertex3f(0, 0, 0);
	drawLine(start, end);

	
	gluLookat(endLine.x, endLine.y, endLine.z, 0, 0, 0, endLine.x - coneUp.x, endLine.y - coneUp.y, endLine.z - coneUp.z);
	glutSolidCone(1, 2, 25, 25);
	
	glPopMatrix();
	*/

	glutSwapBuffers();

	angle += 1.0f / 60.0f;
	speed += 1.0f / 60.0f;
}

void mouseMotion(int x, int y)
{
	float deltaX = x - oldX;
	float deltaY = y - oldY;

	if (motionState == ROTATE) {
		theta -= 0.01f * deltaY;

		if (theta < 0.01f) theta = 0.01f;
		else if (theta > PI / 2.0f - 0.01f) theta = PI / 2.0f - 0.01f;

		phi += 0.01f * deltaX;
		if (phi < 0) phi += 2 * PI;
		else if (phi > 2 * PI) phi -= 2 * PI;
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
	glutCreateWindow("Transformationen");

	glutDisplayFunc(display);
	glutMotionFunc(mouseMotion);
	glutMouseFunc(mouse);
	glutIdleFunc(idle);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glEnable(GL_DEPTH_TEST);

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutMainLoop();
	return 0;
}