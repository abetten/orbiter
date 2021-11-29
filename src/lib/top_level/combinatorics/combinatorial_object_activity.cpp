/*
 * combinatorial_object_activity.cpp
 *
 *  Created on: Mar 20, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



combinatorial_object_activity::combinatorial_object_activity()
{
	Descr = NULL;

	f_has_COC = FALSE;
	COC = NULL;

	f_has_IS = FALSE;
	IS = NULL;

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
	f_has_COC = TRUE;
	combinatorial_object_activity::COC = COC;

	if (f_v) {
		cout << "combinatorial_object_activity::init done" << endl;
	}
}



void combinatorial_object_activity::init_input_stream(combinatorial_object_activity_description *Descr,
		data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::init_input_stream" << endl;
	}

	combinatorial_object_activity::Descr = Descr;
	f_has_IS = TRUE;
	combinatorial_object_activity::IS = IS;


	if (f_v) {
		cout << "combinatorial_object_activity::init_input_stream done" << endl;
	}
}

void combinatorial_object_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity" << endl;
	}
	if (f_has_COC) {
		perform_activity_COC(verbose_level);
	}
	else if (f_has_IS) {
		perform_activity_IS(verbose_level);
	}
	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity done" << endl;
	}
}


void combinatorial_object_activity::perform_activity_COC(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity_COC" << endl;
	}

	if (Descr->f_line_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_COC f_line_type" << endl;
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
			cout << "combinatorial_object_activity::perform_activity_COC line type:" << endl;
			T.print(TRUE /* f_backwards*/);
			cout << endl;
		}


	}

	if (Descr->f_conic_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_COC f_conic_type" << endl;
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
			cout << "combinatorial_object_activity::perform_activity_COC f_conic_type" << endl;
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
			cout << "combinatorial_object_activity::perform_activity_COC f_ideal" << endl;
		}

		projective_space *P;
		homogeneous_polynomial_domain *HPD;

		P = COC->Descr->P;

		HPD = NEW_OBJECT(homogeneous_polynomial_domain);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_COC before HPD->init" << endl;
		}
		HPD->init(P->F, P->n + 1, Descr->ideal_degree,
			FALSE /* f_init_incidence_structure */,
			t_PART /*Monomial_ordering_type*/,
			verbose_level - 2);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_COC after HPD->init" << endl;
		}

		int *Kernel;
		//int *w1, *w2;
		int r;

		Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());
		//w1 = NEW_int(HPD->get_nb_monomials());
		//w2 = NEW_int(HPD->get_nb_monomials());


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_COC the input set is:" << endl;
			HPD->get_P()->print_set_numerical(cout, COC->Pts, COC->nb_pts);
		}


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_COC before HPD->vanishing_ideal" << endl;
		}
		HPD->vanishing_ideal(COC->Pts, COC->nb_pts,
				r, Kernel, verbose_level - 1);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_COC after HPD->vanishing_ideal" << endl;
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
		cout << "combinatorial_object_activity::perform_activity_COC done" << endl;
	}
}


void combinatorial_object_activity::perform_activity_IS(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity_IS" << endl;
	}

	if (Descr->f_canonical_form_PG) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_IS f_canonical_form_PG" << endl;
		}


		int idx;

		idx = Orbiter->find_symbol(Descr->canonical_form_PG_PG_label);

		symbol_table_object_type t;

		t = Orbiter->get_object_type(idx);
		if (t != t_projective_space) {
			cout << "combinatorial_object_activity::perform_activity_IS "
				<< Descr->canonical_form_PG_PG_label << " is not of type projective_space" << endl;
			exit(1);
		}

		projective_space_with_action *PA;

		PA = (projective_space_with_action *) Orbiter->get_object(idx);


		projective_space_object_classifier *OC;

		OC = NEW_OBJECT(projective_space_object_classifier);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_IS before OC->do_the_work" << endl;
		}
		OC->do_the_work(
				Descr->Canonical_form_PG_Descr,
				TRUE,
				PA,
				IS,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_IS after OC->do_the_work" << endl;
		}

		FREE_OBJECT(OC);



	}

}




}}




