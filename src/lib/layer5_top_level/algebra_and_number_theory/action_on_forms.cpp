/*
 * action_on_forms.cpp
 *
 *  Created on: Oct 23, 2022
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;
using namespace orbiter::layer1_foundations;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {



action_on_forms::action_on_forms()
{
	Descr = NULL;

	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;

	q = 0;
	F = NULL;

	f_semilinear = FALSE;

	PA = NULL;

	PF = NULL;

	A_on_poly = NULL;

	f_has_group = FALSE;
	Sg = NULL;

}


action_on_forms::~action_on_forms()
{
}


void action_on_forms::create_action_on_forms(
		action_on_forms_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms::create_action_on_forms" << endl;
	}


	action_on_forms::Descr = Descr;

	if (!Descr->f_space) {
		cout << "action_on_forms::create_action_on_forms please use "
				"-space <space> to specify the projective space" << endl;
		exit(1);
	}
	PA = Get_object_of_projective_space(Descr->space_label);


	F = PA->F;

	PF = NEW_OBJECT(combinatorics::polynomial_function_domain);

	if (f_v) {
		cout << "action_on_forms::create_action_on_forms "
				"before PF->init" << endl;
	}
	PF->init(F, PA->P->n, verbose_level);
	if (f_v) {
		cout << "action_on_forms::create_action_on_forms "
				"after PF->init" << endl;
	}



	//A_on_poly = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "action_on_forms::create_action_on_forms "
				"before PA->A->Induced_action->induced_action_on_homogeneous_polynomials" << endl;
	}


	ring_theory::homogeneous_polynomial_domain *HPD;

	HPD = &PF->Poly[PF->max_degree];

	if (f_v) {
		cout << "action_on_forms::create_action_on_forms "
				"before PA->A->Induced_action->induced_action_on_homogeneous_polynomials" << endl;
	}
	A_on_poly = PA->A->Induced_action->induced_action_on_homogeneous_polynomials(
			HPD,
		FALSE /* f_induce_action */, NULL,
		verbose_level - 2);
	if (f_v) {
		cout << "action_on_forms::create_action_on_forms "
				"after PA->A->Induced_action->induced_action_on_homogeneous_polynomials" << endl;
	}


	int pt = 0;

	f_has_group = TRUE;

	if (f_v) {
		cout << "action_on_forms::create_action_on_forms "
				"computing stabilizer of point, pt = " << pt << endl;
	}
	Sg = PA->A->Strong_gens->point_stabilizer(pt, verbose_level);
	if (f_v) {
		cout << "action_on_forms::create_action_on_forms "
				"after point_stabilizer" << endl;
	}


	if (f_v) {
		cout << "action_on_forms::create_action_on_forms "
				"after A_on_poly->induced_action_on_homogeneous_polynomials" << endl;
	}


	if (f_v) {
		cout << "action_on_forms::create_action_on_forms done" << endl;
	}

}

void action_on_forms::orbits_on_functions(
		int *The_functions, int nb_functions, int len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms::orbits_on_functions" << endl;
		cout << "action_on_forms::orbits_on_functions nb_functions=" << nb_functions << endl;
		cout << "action_on_forms::orbits_on_functions len=" << len << endl;
	}

	ring_theory::homogeneous_polynomial_domain *HPD;

	HPD = &PF->Poly[PF->max_degree];

	int sz;

	sz = HPD->get_nb_monomials();

	int *The_equations;
	int i;

	The_equations = NEW_int(nb_functions * sz);

	if (f_v) {
		cout << "action_on_forms::orbits_on_functions "
				"converting functions to equations" << endl;
	}

	for (i = 0; i < nb_functions; i++) {

		if ((i % 1000) == 0) {
			cout << "i=" << i << " / " << nb_functions << endl;
		}
		PF->compute_polynomial_representation(
				The_functions + i * len,
				The_equations + i * sz,
				0 /*verbose_level*/);

	}

#if 1
	data_structures::int_matrix *Table_of_equations;

	Table_of_equations = NEW_OBJECT(data_structures::int_matrix);

	if (f_v) {
		cout << "action_on_forms::orbits_on_functions "
				"before Table_of_equations->allocate_and_init" << endl;
	}
	Table_of_equations->allocate_and_init(nb_functions, sz, The_equations);
	if (f_v) {
		cout << "action_on_forms::orbits_on_functions "
				"after Table_of_equations->allocate_and_init" << endl;
	}


	if (f_v) {
		cout << "action_on_forms::orbits_on_functions "
				"before Table_of_equations->sort_rows" << endl;
	}
	Table_of_equations->sort_rows(verbose_level);
	if (f_v) {
		cout << "action_on_forms::orbits_on_functions "
				"after Table_of_equations->sort_rows" << endl;
	}


	{
		int a;

		ofstream f("equations.txt");
		f << "table of equations : original number : associated function" << endl;
		for (i = 0; i < nb_functions; i++) {
			f << setw(3) << i << " : ";
			a = Table_of_equations->perm[i];
			Int_vec_print(f, Table_of_equations->M + a * sz, sz);
			f << " : ";
			Int_vec_print(f, The_functions + i * len, len);
			f << endl;
		}
	}
	cout << "equations written to file equations.txt" << endl;




	{
		int a;

		ofstream f("equations_sorted.txt");
		f << "sorted table of equations : original number : associated function" << endl;
		for (i = 0; i < nb_functions; i++) {
			f << setw(3) << i << " : ";
			Int_vec_print(f, Table_of_equations->M + i * sz, sz);
			f << " : ";
			a = Table_of_equations->perm_inv[i];
			f << a;
			f << " : ";
			Int_vec_print(f, The_functions + a * len, len);
			f << endl;
		}
	}
	cout << "equations written to file equations0.txt" << endl;

	Table_of_equations->check_that_entries_are_distinct(verbose_level);

	if (f_v) {
		cout << "action_on_forms::orbits_on_functions "
				"sorted Table_of_equations:" << endl;
		//Table_of_equations->print();
	}
#endif

	actions::action *A_on_equations;
	groups::schreier *Orb;

	if (f_v) {
		cout << "action_on_forms::orbits_on_functions "
				"computing orbits on equations" << endl;
	}

	orbits_on_equations(The_equations, nb_functions, sz, Orb, A_on_equations, verbose_level);

	if (f_v) {
		cout << "action_on_forms::orbits_on_functions "
				"computing orbits on equations finished" << endl;
	}


	data_structures::tally *Classify_orbits_by_length;

	if (f_v) {
		cout << "action_on_forms::orbits_on_functions" << endl;
	}
	Classify_orbits_by_length = NEW_OBJECT(data_structures::tally);
	Classify_orbits_by_length->init(Orb->orbit_len, Orb->nb_orbits, FALSE, 0);


	data_structures::set_of_sets *SoS;
	int *types;
	int nb_types;

	SoS = Classify_orbits_by_length->get_set_partition_and_types(types,
			nb_types, verbose_level);
	if (f_v) {
		cout << "action::orbits_on_equations "
				"We found the following types:" << endl;
		Int_vec_print(cout, types, nb_types);
		cout << endl;
		cout << "action::orbits_on_equations "
				"Orbits of type 0: ";
		Lint_vec_print(cout, SoS->Sets[0], SoS->Set_size[0]);
		cout << endl;
	}



	if (f_v) {
		cout << "action::orbits_on_equations "
				"We found " << Orb->nb_orbits
				<< " orbits on the equations. "
						"The non-trivial orbits are:" << endl;
		Orb->print_and_list_non_trivial_orbits_tex(cout);

		cout << "action::orbits_on_equations "
				"The orbits that are not long are:" << endl;
		int h, t, t_max;

		if (nb_types > 1) {
			t_max = nb_types - 1;
		}
		else {
			t_max = nb_types;
		}
		for (t = 0; t < t_max; t++) {

			// skip the long orbits

			cout << "Type " << t << ":" << endl;
			cout << "There are " << SoS->Set_size[t]
				<< " orbits of length " << types[t] << ":" << endl;

			for (h = 0; h < SoS->Set_size[t]; h++) {
				Orb->print_and_list_orbit_tex(SoS->Sets[t][h], cout);
			}
			cout << endl;
			for (h = 0; h < SoS->Set_size[t]; h++) {
				int *v;
				int orbit_len;
				int u, orbit_element;

				cout << "orbit " << h << " / " << SoS->Set_size[t] << " of length " << types[t]
						<< " is orbit " << SoS->Sets[t][h] << ":" << endl;
				Orb->print_and_list_orbit_tex(SoS->Sets[t][h], cout);
				Orb->get_orbit_sorted(v, orbit_len, SoS->Sets[t][h]);

				for (u = 0; u < orbit_len; u++) {

					data_structures::int_matrix *Table_of_equations;

					orbit_element = v[u];
					cout << orbit_element << " = ";
					Table_of_equations = A_on_equations->G.OnHP->Table_of_equations;

					int *coeffs;

					coeffs = Table_of_equations->M +
							Table_of_equations->perm[orbit_element] * Table_of_equations->n;

					Int_vec_print(cout,
							coeffs,
							Table_of_equations->n);

					cout << " = ";

					HPD->print_equation_tex(cout, coeffs);

					cout << " = ";

					Int_vec_print(cout, The_functions + orbit_element * len, len);

					cout << endl;
				}
			}
		}

	}

	if (f_v) {
		cout << "action_on_forms::orbits_on_functions "
				"The distribution of orbit lengths is: ";
		Classify_orbits_by_length->print_naked(FALSE);
		cout << endl;
	}



	FREE_int(The_equations);

	if (f_v) {
		cout << "action_on_forms::orbits_on_functions done" << endl;
	}
}

void action_on_forms::orbits_on_equations(
		int *The_equations, int nb_equations, int len,
		groups::schreier *&Orb,
		actions::action *&A_on_equations,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms::orbits_on_equations" << endl;
		cout << "action_on_forms::orbits_on_equations nb_equations=" << nb_equations << endl;
		cout << "action_on_forms::orbits_on_equations len=" << len << endl;
	}


	ring_theory::homogeneous_polynomial_domain *HPD;

	HPD = &PF->Poly[PF->max_degree];

	actions::action_global AG;


	AG.orbits_on_equations(
			PA->A,
			HPD,
			The_equations, nb_equations, Sg,
			A_on_equations,
			Orb, verbose_level);


	if (f_v) {
		cout << "action_on_forms::orbits_on_equations done" << endl;
	}
}

void action_on_forms::associated_set_in_plane(
		int *func, int len,
		long int *&Rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms::associated_set_in_plane" << endl;
	}

	if (len != PF->Q) {
		cout << "action_on_forms::associated_set_in_plane "
				"len should be " << PF->Q << endl;
		cout << "but is equal to " << len << endl;
		exit(1);
	}

	int x, y;
	int v[3];
	long int a;

	Rk = NEW_lint(len);

	for (x = 0; x < len; x++) {
		y = func[x];
		v[0] = x;
		v[1] = y;
		v[2] = 1;
		F->Projective_space_basic->PG_element_rank_modified_lint(
				v, 1, 3, a);
		Rk[x] = a;
	}
	if (f_v) {
		cout << "action_on_forms::associated_set_in_plane "
				"associated point set: ";
		Lint_vec_print(cout, Rk, len);
		cout << endl;
	}


	if (f_v) {
		cout << "action_on_forms::create_set_in_plane done" << endl;
	}
}


void action_on_forms::differential_uniformity(
		int *func, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_forms::differential_uniformity" << endl;
	}

	if (len != PF->Q) {
		cout << "action_on_forms::differential_uniformity "
				"len should be " << PF->Q << endl;
		cout << "but is equal to " << len << endl;
		exit(1);
	}

	int *nb_times_ab;

	if (f_v) {
		cout << "algebra_global::search_APN" << endl;
	}
	q = F->q;

	nb_times_ab = NEW_int(q * q);

	std::vector<std::vector<int> > Solutions;

	Int_vec_zero(nb_times_ab, q * q);

	//algebra::algebra_global Algebra;
	int delta;
	int *Fibre;

	combinatorics::apn_functions *Apn_functions;


	Apn_functions = NEW_OBJECT(combinatorics::apn_functions);

	Apn_functions->init(F, verbose_level);


	delta = Apn_functions->differential_uniformity(
			func, nb_times_ab, 0 /* verbose_level */);


	delta = Apn_functions->differential_uniformity_with_fibre(
			func, nb_times_ab, Fibre, verbose_level);


	if (f_v) {
		cout << "action_on_forms::differential_uniformity "
				"nb_times_ab: " << endl;
		Int_matrix_print(nb_times_ab, q, q);
		cout << endl;
		cout << "action_on_forms::differential_uniformity "
				"delta = " << delta << endl;

		int a, b, nb, x, h;

		if (delta == 1) {
			for (a = 1; a < q; a++) {
				cout << "a = " << a << " : ";
				for (b = 0; b < q; b++) {
					x = Fibre[(a * q + b) * delta + 0];
					cout << x;
					if (b < q - 1) {
						cout << ", ";
					}
				}
				cout << endl;
			}
		}
		else {
			for (a = 0; a < q; a++) {
				for (b = 0; b < q; b++) {
					nb = nb_times_ab[a * q + b];
					for (h = 0; h < nb; h++) {
						x = Fibre[(a * q + b) * delta + h];
						cout << a << " : " << b << " : " << x << endl;
					}
				}
			}

		}
	}

	FREE_OBJECT(Apn_functions);


	if (f_v) {
		cout << "action_on_forms::differential_uniformity done" << endl;
	}
}



}}}



