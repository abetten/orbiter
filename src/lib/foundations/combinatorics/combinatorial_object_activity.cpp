/*
 * combinatorial_object_activity.cpp
 *
 *  Created on: Mar 20, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


combinatorial_object_activity::combinatorial_object_activity()
{
	Descr = NULL;
	COC = NULL;

}

combinatorial_object_activity::~combinatorial_object_activity()
{
}


void combinatorial_object_activity::init(combinatorial_object_activity_description *Descr,
		combinatorial_object_create *COC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::init" << endl;
	}

	combinatorial_object_activity::Descr = Descr;
	combinatorial_object_activity::COC = COC;


	if (f_v) {
		cout << "combinatorial_object_activity::init done" << endl;
	}
}

void combinatorial_object_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity" << endl;
	}


	if (Descr->f_line_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity f_line_type" << endl;
		}

		projective_space *P;

		P = COC->Descr->P;

		int *type;

		type = NEW_int(P->N_lines);


		P->line_intersection_type(
				COC->Pts, COC->nb_pts, type, 0 /* verbose_level */);
			// type[N_lines]


		tally T;

		T.init(type, P->N_lines, FALSE, 0);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity line type:" << endl;
			T.print(TRUE /* f_backwards*/);
			cout << endl;
		}


	}

	if (Descr->f_conic_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity f_conic_type" << endl;
		}

		projective_space *P;

		P = COC->Descr->P;

		long int **Pts_on_conic;
		int **Conic_eqn;
		int *nb_pts_on_conic;
		int len;
		int i;

		P->conic_type(
				COC->Pts, COC->nb_pts,
				Descr->conic_type_threshold,
				Pts_on_conic, Conic_eqn, nb_pts_on_conic, len,
				verbose_level);


		cout << "We found " << len << " conics" << endl;
		for (i = 0; i < len; i++) {
			cout << i << " : " << nb_pts_on_conic << endl;
		}
		//

	}

	if (Descr->f_non_conical_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity f_conic_type" << endl;
		}

		projective_space *P;

		P = COC->Descr->P;

		std::vector<int> Rk;

		P->determine_nonconical_six_subsets(
				COC->Pts, COC->nb_pts,
				Rk,
				verbose_level);

		cout << "We found " << Rk.size() << " non-conical 6 subsets" << endl;

	}


	if (Descr->f_ideal) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity f_ideal" << endl;
		}

		projective_space *P;
		homogeneous_polynomial_domain *HPD;

		P = COC->Descr->P;

		HPD = NEW_OBJECT(homogeneous_polynomial_domain);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity before HPD->init" << endl;
		}
		HPD->init(P->F, P->n + 1, Descr->ideal_degree,
			FALSE /* f_init_incidence_structure */,
			t_PART /*Monomial_ordering_type*/,
			verbose_level - 2);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity after HPD->init" << endl;
		}

		int *Kernel;
		//int *w1, *w2;
		int r;

		Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());
		//w1 = NEW_int(HPD->get_nb_monomials());
		//w2 = NEW_int(HPD->get_nb_monomials());


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity the input set is:" << endl;
			HPD->get_P()->print_set_numerical(cout, COC->Pts, COC->nb_pts);
		}


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity before HPD->vanishing_ideal" << endl;
		}
		HPD->vanishing_ideal(COC->Pts, COC->nb_pts,
				r, Kernel, verbose_level - 1);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity after HPD->vanishing_ideal" << endl;
		}

		int h, ns;
		int nb_pts;
		long int *Pts;

		ns = HPD->get_nb_monomials() - r; // dimension of null space

		cout << "looping over all generators of the ideal:" << endl;
		for (h = 0; h < ns; h++) {
			cout << "generator " << h << " / " << ns << " is ";
			Orbiter->Int_vec.print(cout, Kernel + h * HPD->get_nb_monomials(), HPD->get_nb_monomials());
			cout << " : " << endl;

			vector<long int> Points;
			int i;

			HPD->enumerate_points(Kernel + h * HPD->get_nb_monomials(),
					Points, verbose_level);
			nb_pts = Points.size();

			Pts = NEW_lint(nb_pts);
			for (i = 0; i < nb_pts; i++) {
				Pts[i] = Points[i];
			}


			cout << "We found " << nb_pts << " points on the generator of the ideal" << endl;
			cout << "They are : ";
			Orbiter->Lint_vec.print(cout, Pts, nb_pts);
			cout << endl;
			HPD->get_P()->print_set_numerical(cout, Pts, nb_pts);

#if 0
			if (h == 0) {
				size_out = HPD->get_nb_monomials();
				set_out = NEW_lint(size_out);
				//int_vec_copy(Kernel + h * HPD->nb_monomials, set_out, size_out);
				int u;
				for (u = 0; u < size_out; u++) {
					set_out[u] = Kernel[h * HPD->get_nb_monomials() + u];
				}
				//break;
			}
#endif
			FREE_lint(Pts);

		} // next h

	}


	if (Descr->f_save) {

		file_io Fio;
		string fname;

		fname.assign(COC->fname);

		if (f_v) {
			cout << "We will write to the file " << fname << endl;
		}
		Fio.write_set_to_file(fname, COC->Pts, COC->nb_pts, verbose_level);
		if (f_v) {
			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity done" << endl;
	}
}

}}


