/*
 * projective_space3.cpp
 *
 *  Created on: Jan 9, 2022
 *      Author: betten
 */




#include "foundations.h"


using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace geometry {


int projective_space::reverse_engineer_semilinear_map(
	int *Elt, int *Mtx, int &frobenius,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//finite_field *F;
	int d = n + 1;
	int *v1, *v2, *v1_save;
	int *w1, *w2, *w1_save;
	int /*q,*/ h, hh, i, j, l, e, frobenius_inv, lambda, rk, c, cv;
	int *system;
	int *base_cols;
	number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "projective_space::reverse_engineer_semilinear_map" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "d=" << d << endl;
	}
	//F = P->F;
	//q = F->q;

	v1 = NEW_int(d);
	v2 = NEW_int(d);
	v1_save = NEW_int(d);
	w1 = NEW_int(d);
	w2 = NEW_int(d);
	w1_save = NEW_int(d);



	if (f_v) {
		cout << "projective_space::reverse_engineer_semilinear_map "
				"mapping unit vectors" << endl;
	}
	for (e = 0; e < d; e++) {
		// map the unit vector e_e
		// (with a one in position e and zeros elsewhere):
		for (h = 0; h < d; h++) {
			if (h == e) {
				v1[h] = 1;
			}
			else {
				v1[h] = 0;
			}
		}
		Int_vec_copy(v1, v1_save, d);
		i = rank_point(v1);
			// Now, the value of i should be equal to e.
		//j = element_image_of(i, Elt, 0);
		j = Elt[i];
		unrank_point(v2, j);
		if (f_v) {
			cout << "projective_space::reverse_engineer_semilinear_map "
					"unit vector " << e << " has rank " << i << " and maps to " << j << endl;
		}

#if 0
		if (f_vv) {
			print_from_to(d, i, j, v1_save, v2);
		}
#endif


		Int_vec_copy(v2, Mtx + e * d, d);
	}

	if (f_vv) {
		cout << "Mtx (before scaling):" << endl;
		Int_vec_print_integer_matrix_width(cout, Mtx, d, d, d, F->log10_of_q);
		cout << endl;
	}

	// map the vector (1,1,...,1):
	if (f_v) {
		cout << "projective_space::reverse_engineer_semilinear_map "
				"mapping the all-one vector"
				<< endl;
	}
	for (h = 0; h < d; h++) {
		v1[h] = 1;
	}
	Int_vec_copy(v1, v1_save, d);
	i = rank_point(v1);
	//j = element_image_of(i, Elt, 0);
	j = Elt[i];
	unrank_point(v2, j);
	if (f_v) {
		cout << "projective_space::reverse_engineer_semilinear_map "
				"the all one vector has rank " << i << " and maps to " << j << endl;
	}

#if 0
	if (f_vv) {
		print_from_to(d, i, j, v1_save, v2);
	}
#endif

	system = NEW_int(d * (d + 1));
	base_cols = NEW_int(d + 1);
	// coefficient matrix:
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			system[i * (d + 1) + j] = Mtx[j * d + i];
		}
	}
	// RHS:
	for (i = 0; i < d; i++) {
		system[i * (d + 1) + d] = v2[i];
	}
	if (f_vv) {
		cout << "projective_space::reverse_engineer_semilinear_map "
				"linear system:" << endl;
		Int_vec_print_integer_matrix_width(cout, system,
				d, d + 1, d + 1, F->log10_of_q);
		cout << endl;
	}
	rk = F->Linear_algebra->Gauss_simple(system, d, d + 1, base_cols, verbose_level - 4);
	if (rk != d) {
		cout << "projective_space::reverse_engineer_semilinear_map "
				"rk != d, fatal" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "projective_space::reverse_engineer_semilinear_map "
				"after Gauss_simple:" << endl;
		Int_vec_print_integer_matrix_width(cout, system,
				d, d + 1, d + 1, F->log10_of_q);
		cout << endl;
	}
	for (i = 0; i < d; i++) {
		c = system[i * (d + 1) + d];
		if (c == 0) {
			cout << "projective_space::reverse_engineer_semilinear_map "
					"the input matrix does not have full rank" << endl;
			exit(1);
		}
		for (j = 0; j < d; j++) {
			Mtx[i * d + j] = F->mult(c, Mtx[i * d + j]);
		}
	}

	if (f_vv) {
		cout << "Mtx (after scaling):" << endl;
		Int_vec_print_integer_matrix_width(cout, Mtx, d, d, d, F->log10_of_q);
		cout << endl;
	}



	frobenius = 0;
	if (F->q != F->p) {

		// figure out the frobenius:
		if (f_v) {
			cout << "projective_space::reverse_engineer_semilinear_map "
					"figuring out the frobenius" << endl;
		}


		// create the vector (1,p,0,...,0)

		for (h = 0; h < d; h++) {
			if (h == 0) {
				v1[h] = 1;
			}
			else if (h == 1) {
				v1[h] = F->p;
			}
			else {
				v1[h] = 0;
			}
		}
		Int_vec_copy(v1, v1_save, d);
		i = rank_point(v1);
		//j = element_image_of(i, Elt, 0);
		j = Elt[i];
		unrank_point(v2, j);


#if 0
		if (f_vv) {
			print_from_to(d, i, j, v1_save, v2);
		}
#endif


		// coefficient matrix:
		for (i = 0; i < d; i++) {
			for (j = 0; j < 2; j++) {
				system[i * 3 + j] = Mtx[j * d + i];
			}
		}
		// RHS:
		for (i = 0; i < d; i++) {
			system[i * 3 + 2] = v2[i];
		}
		rk = F->Linear_algebra->Gauss_simple(system,
				d, 3, base_cols, verbose_level - 4);
		if (rk != 2) {
			cout << "rk != 2, fatal" << endl;
			exit(1);
		}
		if (f_vv) {
			cout << "after Gauss_simple:" << endl;
			Int_vec_print_integer_matrix_width(cout,
					system, 2, 3, 3, F->log10_of_q);
			cout << endl;
		}

		c = system[0 * 3 + 2];
		if (c != 1) {
			cv = F->inverse(c);
			for (hh = 0; hh < 2; hh++) {
				system[hh * 3 + 2] = F->mult(cv, system[hh * 3 + 2]);
			}
		}
		if (f_vv) {
			cout << "after scaling the last column:" << endl;
			Int_vec_print_integer_matrix_width(cout,
					system, 2, 3, 3, F->log10_of_q);
			cout << endl;
		}
		lambda = system[1 * 3 + 2];
		if (f_vv) {
			cout << "lambda=" << lambda << endl;
		}


		l = F->log_alpha(lambda);
		if (f_vv) {
			cout << "l=" << l << endl;
		}
		for (i = 0; i < F->e; i++) {
			if (NT.i_power_j(F->p, i) == l) {
				frobenius = i;
				break;
			}
		}
		if (i == F->e) {
			cout << "projective_space::reverse_engineer_semilinear_map "
					"problem figuring out the Frobenius" << endl;
			exit(1);
		}

		frobenius_inv = (F->e - frobenius) % F->e;
		if (f_vv) {
			cout << "frobenius = " << frobenius << endl;
			cout << "frobenius_inv = " << frobenius_inv << endl;
		}
		for (hh = 0; hh < d * d; hh++) {
			Mtx[hh] = F->frobenius_power(Mtx[hh], frobenius_inv);
		}


	}
	else {
		frobenius = 0;
		frobenius_inv = 0;
	}


	if (f_v) {
		cout << "projective_space::reverse_engineer_semilinear_map "
				"we found the following map" << endl;
		cout << "Mtx:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Mtx, d, d, d, F->log10_of_q);
		cout << endl;
		cout << "frobenius = " << frobenius << endl;
		cout << "frobenius_inv = " << frobenius_inv << endl;
	}



	FREE_int(v1);
	FREE_int(v2);
	FREE_int(v1_save);
	FREE_int(w1);
	FREE_int(w2);
	FREE_int(w1_save);
	FREE_int(system);
	FREE_int(base_cols);

	if (f_v) {
		cout << "projective_space::reverse_engineer_semilinear_map done" << endl;
	}

	return TRUE;
}

void projective_space::planes_through_line(long int *Lines, int nb_lines,
		long int *&Plane_ranks, int &nb_planes_on_one_line, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, d;
	int *M;

	if (f_v) {
		cout << "projective_space::planes_through_line" << endl;
	}

	d = n + 1;
	M = NEW_int(3 * d);

	for (i = 0; i < nb_lines; i++) {

		std::vector<long int> plane_ranks;

		planes_through_a_line(
				Lines[i], plane_ranks,
				0 /*verbose_level*/);

		nb_planes_on_one_line = plane_ranks.size();

		if (i == 0) {
			Plane_ranks = NEW_lint(nb_lines * nb_planes_on_one_line);
		}
		for (j = 0; j < plane_ranks.size(); j++) {
			Plane_ranks[i * nb_planes_on_one_line + j] = plane_ranks[j];
		}

		if (f_v) {
			cout << "planes through line " << Lines[i] << " : ";
			for (j = 0; j < plane_ranks.size(); j++) {
				cout << plane_ranks[j];
				if (j < plane_ranks.size() - 1) {
					cout << ",";
				}
			}
			cout << endl;

			cout << "planes through line " << Lines[i] << endl;
			for (j = 0; j < plane_ranks.size(); j++) {
				cout << j << " : " << plane_ranks[j] << " : " << endl;
				Grass_planes->unrank_lint_here(M, plane_ranks[j], 0 /* verbose_level */);
				Int_matrix_print(M, 3, d);

			}
			cout << endl;
		}


	}
	FREE_int(M);
	if (f_v) {
		cout << "projective_space::planes_through_line done" << endl;
	}

}



}}}

