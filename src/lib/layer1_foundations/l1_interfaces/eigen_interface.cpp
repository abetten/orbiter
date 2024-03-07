/*
 * eigen_interface.cpp
 *
 *  Created on: Dec 18, 2021
 *      Author: betten
 */


#include "foundations.h"


#include <../Eigen_interface/Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace l1_interfaces {


void orbiter_eigenvalues(
		int *Mtx, int nb_points, double *E, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbiter_eigenvalues" << endl;
	}


#if 0
	MatrixXd X = MatrixXd::Random(5,5);
	MatrixXd A = X + X.transpose();
	cout << "Here is a random symmetric 5x5 matrix, A:" << endl << A << endl << endl;

	SelfAdjointEigenSolver<MatrixXd> es(A);
	cout << "The eigenvalues of A are:" << endl << es.eigenvalues() << endl;
	cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;

	double lambda = es.eigenvalues()[0];
	cout << "Consider the first eigenvalue, lambda = " << lambda << endl;
	VectorXd v = es.eigenvectors().col(0);
	cout << "If v is the corresponding eigenvector, then lambda * v = " << endl << lambda * v << endl;
	cout << "... and A * v = " << endl << A * v << endl << endl;

	MatrixXd D = es.eigenvalues().asDiagonal();
	MatrixXd V = es.eigenvectors();
	cout << "Finally, V * D * V^(-1) = " << endl << V * D * V.inverse() << endl;
#else

	int i, j;

	MatrixXd X(nb_points, nb_points);

	for (i = 0; i < nb_points; i++) {
		X(i, i) = 0;
	}
	for (i = 0; i < nb_points; i++) {
		for (j = 0; j < nb_points; j++) {
			X(i, j) = Mtx[i * nb_points + j];
		}
	}

	SelfAdjointEigenSolver<MatrixXd> es(X);

	if (f_v) {
		cout << "The eigenvalues of X are:" << endl << es.eigenvalues() << endl;
		cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;
	}

	for (i = 0; i < nb_points; i++) {
		E[i] = es.eigenvalues()[i];
	}

#endif

	if (f_v) {
		cout << "orbiter_eigenvalues done" << endl;
	}
}


}}}

