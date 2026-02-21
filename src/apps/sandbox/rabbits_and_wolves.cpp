/*
 * rabbits_and_wolves.cpp
 *
 *  Created on: Feb 21, 2026
 *      Author: betten
 */


#include <iostream>
#include <fstream>

#include <math.h>




using namespace std;




void rabbit_and_wolves_euler_step(
		double xi, double yi, double &xip1, double &yip1, double h)
{
	double k = 0.08;
	double z = 0.0002;
	double a = 0.001;
	double r = 0.02;
	double b = 0.00002;
	double xp, yp;

	xp = k * xi * (1. - z * xi) - a * xi * yi;
	yp = -r * yi + b * xi * yi;

	xip1 = xi + h * xp;
	yip1 = yi + h * yp;
}

int main()
{

	double *X, *Y;
	int N;
	double x0 = 50;
	double y0 = 10;
	int i;
	double h = 1.0;

	N = 1000;
	X = new double[N + 1];
	Y = new double[N + 1];

	{
	ofstream ost("data.txt");

	ost << "# data.txt" << endl;
	ost << "# X Y" << endl;


	X[0] = x0;
	Y[0] = y0;
	for (i = 1; i <= N; i++) {
		rabbit_and_wolves_euler_step(
				X[i - 1], Y[i - 1], X[i], Y[i], h);
		cout << i << " : " << X[i] << ", " << Y[i] << endl;
		ost << X[i] << ", " << Y[i] << endl;
	}
	cout << "X[" << N << "]=" << X[N] << endl;
	cout << "Y[" << N << "]=" << Y[N] << endl;

	}
}



