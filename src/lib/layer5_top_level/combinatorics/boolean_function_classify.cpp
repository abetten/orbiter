/*
 * boolean_function_classify.cpp
 *
 *  Created on: Nov 06, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


static void boolean_function_classify_print_function(
		int *poly, int sz, void *data);
static void boolean_function_classify_reduction_function(
		int *poly, void *data);


boolean_function_classify::boolean_function_classify()
{
	BF = NULL;
	A = NULL;
	//nice_gens = NULL;
	AonHPD = NULL;
	SG = NULL;
	A_affine = NULL;
}

boolean_function_classify::~boolean_function_classify()
{
#if 0
	if (A) {
		FREE_OBJECT(A);
	}
	if (nice_gens) {
		FREE_OBJECT(nice_gens);
	}
#endif
	if (AonHPD) {
		FREE_OBJECT(AonHPD);
	}
	if (SG) {
		FREE_OBJECT(SG);
	}
	if (A_affine) {
		FREE_OBJECT(A_affine);
	}
}



void boolean_function_classify::init_group(
		combinatorics::boolean_function_domain *BF,
		actions::action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "boolean_function_classify::init_group" << endl;
	}

	boolean_function_classify::BF = BF;

	int degree = BF->n + 1;

	boolean_function_classify::A = A;

	AonHPD = NEW_OBJECT(induced_actions::action_on_homogeneous_polynomials);
	if (f_v) {
		cout << "boolean_function_classify::init_group "
				"before AonHPD->init" << endl;
	}
	AonHPD->init(A, &BF->Poly[BF->n], verbose_level);
	if (f_v) {
		cout << "boolean_function_classify::init_group "
				"after AonHPD->init" << endl;
	}


	SG = NEW_OBJECT(groups::strong_generators);

	algebra::matrix_group *Mtx;

	Mtx = A->get_matrix_group();

	if (f_v) {
		cout << "boolean_function_classify::init_group "
				"before generators_for_parabolic_subgroup" << endl;
	}
	SG->generators_for_parabolic_subgroup(A,
			Mtx, degree - 1, verbose_level);
	if (f_v) {
		cout << "boolean_function_classify::init_group "
				"after generators_for_parabolic_subgroup" << endl;
	}

	SG->print_generators_tex(cout);

	SG->group_order(go);
	if (f_v) {
		cout << "boolean_function_classify::init_group "
				"go=" << go << endl;
	}

	std::string label_of_set;


	label_of_set.assign("affine_points");

	if (f_v) {
		cout << "boolean_function_classify::init_group "
				"before A->Induced_action->restricted_action" << endl;
	}
	A_affine = A->Induced_action->restricted_action(
			BF->affine_points, BF->Q, label_of_set,
			verbose_level);
	if (f_v) {
		cout << "boolean_function_classify::init_group "
				"after A->Induced_action->restricted_action" << endl;
	}

	if (f_v) {
		cout << "Generators in the induced action:" << endl;
		SG->print_with_given_action(
			cout, A_affine);
	}


#if 0
	SG->init(A);
	if (f_v) {
		cout << "boolean_function::init_group "
				"before init_transposed_group" << endl;
	}
	SG->init_transposed_group(SGt, verbose_level);
	if (f_v) {
		cout << "boolean_function::init_group "
				"after init_transposed_group" << endl;
	}
#endif

	if (f_v) {
		cout << "boolean_function_classify::init_group done" << endl;
	}
}


void boolean_function_classify::search_for_bent_functions(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *poly;
	int i, j;
	ring_theory::longinteger_object a;
	int nb_sol = 0;
	int nb_orbits = 0;
	uint32_t h;
	geometry::geometry_global Gg;
	ring_theory::longinteger_domain D;
	data_structures::data_structures_global Data;
	vector<int> orbit_first;
	vector<int> orbit_length;

	vector<vector<int> > Bent_function_table;
	vector<vector<int> > Equation_table;

	std::multimap<uint32_t, int> Hashing;
		// we store the pair (hash, idx)
		// where hash is the hash value of the set and idx is the
		// index in the table Sets where the set is stored.
		//
		// we use a multimap because the hash values are not unique
		// it happens that two sets have the same hash value.
		// map cannot handle that.


	if (f_v) {
		cout << "boolean_function_classify::search_for_bent_functions" << endl;
	}



	poly = NEW_int(BF->Poly[BF->n].get_nb_monomials());

	a.create(0);
	while (D.is_less_than(a, *BF->NN)) {

		Gg.AG_element_unrank_longinteger(2, BF->f, 1, BF->Q, a);
		//Gg.AG_element_unrank(2, f, 1, Q, a);
		cout << a << " / " << BF->NN << " : ";
		Int_vec_print(cout, BF->f, BF->Q);
		//cout << endl;

		BF->raise(BF->f, BF->F);

		BF->apply_Walsh_transform(BF->F, BF->T);

		cout << " : ";
		Int_vec_print(cout, BF->T, BF->Q);

		if (BF->is_bent(BF->T)) {
			cout << " is bent " << nb_sol;
			nb_sol++;

			h = Data.int_vec_hash(BF->f, BF->Q);

		    map<uint32_t, int>::iterator itr, itr1, itr2;
		    int pos, f_found;

		    itr1 = Hashing.lower_bound(h);
		    itr2 = Hashing.upper_bound(h);
		    f_found = false;
		    for (itr = itr1; itr != itr2; ++itr) {
		        pos = itr->second;
		        for (j = 0; j < BF->Q; j++) {
		        	if (BF->f[j] != Bent_function_table[pos][j]) {
		        		break;
		        	}
		        }
		        if (j == BF->Q) {
		        	f_found = true;
		        	break;
		        }
		    }


		    if (!f_found) {

				cout << " NEW orbit " << nb_orbits << endl;

				BF->compute_polynomial_representation(BF->f, poly, 0 /*verbose_level*/);
				cout << " : ";
				BF->Poly[BF->n].print_equation(cout, poly);
				cout << " : ";
				//evaluate_projectively(poly, f_proj);
				BF->evaluate(poly, BF->f_proj);
				Int_vec_print(cout, BF->f_proj, BF->Q);
				cout << endl;

				orbits_schreier::orbit_of_equations *Orb;

				Orb = NEW_OBJECT(orbits_schreier::orbit_of_equations);

				Orb->f_has_print_function = true;
				Orb->print_function = boolean_function_classify_print_function;
				Orb->print_function_data = this;

				Orb->f_has_reduction = true;
				Orb->reduction_function = boolean_function_classify_reduction_function;
				Orb->reduction_function_data = this;

				cout << "orbit " << nb_orbits << ", computing orbit of bent function:" << endl;
				Orb->init(A, BF->Fq,
					AonHPD,
					SG /* A->Strong_gens*/, poly,
					0 /*verbose_level*/);
				cout << "found an orbit of length " << Orb->used_length << endl;

				groups::strong_generators *Stab_gens;

				cout << "orbit " << nb_orbits << ", computing stabilizer:" << endl;
				Stab_gens = Orb->stabilizer_orbit_rep(
						go, verbose_level);
				Stab_gens->print_generators_tex(cout);

				orbit_first.push_back(Bent_function_table.size());
				orbit_length.push_back(Orb->used_length);

				int *coeff;

				for (i = 0; i < Orb->used_length; i++) {
					coeff = Orb->Equations[i] + 1;
					BF->evaluate(coeff, BF->f_proj);
					vector<int> v;
					for (j = 0; j < BF->Q; j++) {
						v.push_back(BF->f_proj[j]);
					}
					vector<int> w;
					for (j = 0; j < BF->Poly[BF->n].get_nb_monomials(); j++) {
						w.push_back(Orb->Equations[i][1 + j]);
					}

					h = Data.int_vec_hash(BF->f_proj, BF->Q);
					Hashing.insert(pair<uint32_t, int>(h, Bent_function_table.size()));

					Bent_function_table.push_back(v);
					Equation_table.push_back(w);
				}

				//int idx = 3;
				int idx = 0;

				if (BF->n == 4) {
					if (nb_orbits == 0) {
						idx = 12;
					}
					else if (nb_orbits == 1) {
						idx = 180;
					}
				}

				if (Orb->used_length > idx) {
					cout << "orbit " << nb_orbits << ", computing stabilizer of element " << idx << endl;

					coeff = Orb->Equations[idx] + 1;
					BF->evaluate(coeff, BF->f_proj);
					cout << "orbit " << nb_orbits << ", function: ";
					Int_vec_print(cout, BF->f_proj, BF->Q);
					cout << endl;
					cout << "orbit " << nb_orbits << ", equation: ";
					Int_vec_print(cout, coeff, BF->Poly[BF->n].get_nb_monomials());
					cout << endl;

					groups::strong_generators *Stab_gens_clean;


					Stab_gens_clean = Orb->stabilizer_any_point(
							go, idx,
							verbose_level);
					Stab_gens_clean->print_generators_tex(cout);
					cout << "orbit " << nb_orbits << ", induced action:" << endl;
					Stab_gens_clean->print_with_given_action(
							cout, A_affine);

					FREE_OBJECT(Stab_gens_clean);
				}

				FREE_OBJECT(Stab_gens);
				FREE_OBJECT(Orb);

				nb_orbits++;

		    }
		    else {
		    	cout << "The bent function has been found earlier already" << endl;
		    }
		}
		else {
			cout << endl;
		}
		a.increment();
		cout << "after increment: a=" << a << endl;
	}
	cout << "We found " << nb_sol << " bent functions" << endl;
	cout << "We have " << Bent_function_table.size() << " bent functions in the table" << endl;
	cout << "They fall into " << orbit_first.size() << " orbits:" << endl;

	int fst, len, t;

	for (h = 0; h < orbit_first.size(); h++) {
		fst = orbit_first[h];
		len = orbit_length[h];
		cout << "Orbit " << h << " / " << orbit_first.size() << " has length " << len << ":" << endl;
		for (t = 0; t < len; t++) {
			i = fst + t;
			cout << i << " : " << t << " / " << len << " : ";
			for (j = 0; j < BF->Q; j++) {
				BF->f[j] = Bent_function_table[i][j];
			}
			for (j = 0; j < BF->Poly[BF->n].get_nb_monomials(); j++) {
				poly[j] = Equation_table[i][j];
			}

			Int_vec_copy(BF->f, BF->f2, BF->Q);
			Gg.AG_element_rank_longinteger(2, BF->f2, 1, BF->Q, a);

			Int_vec_print(cout, BF->f, BF->Q);
			cout << " : " << a << " : ";
			Int_vec_print(cout, poly, BF->Poly[BF->n].get_nb_monomials());
			cout << " : ";
			BF->Poly[BF->n].print_equation(cout, poly);
			cout << endl;
		}
	}
#if 0
	for (i = 0; i < Bent_function_table.size(); i++) {
		cout << i << " : ";
		for (j = 0; j < Q; j++) {
			f[j] = Bent_function_table[i][j];
		}
		for (j = 0; j < Poly[n].get_nb_monomials(); j++) {
			poly[j] = Equation_table[i][j];
		}
		int_vec_print(cout, f, Q);
		cout << " : ";
		int_vec_print(cout, poly, Poly[n].get_nb_monomials());
		cout << " : ";
		Poly[n].print_equation(cout, poly);
		cout << endl;
	}
#endif



	for (h = 0; h < orbit_first.size(); h++) {
		cout << "orbit " << h << " / " << orbit_first.size() << " has length " << orbit_length[h] << endl;
	}

	FREE_int(poly);

	if (f_v) {
		cout << "boolean_function_classify::search_for_bent_functions done" << endl;
	}
}



static void boolean_function_classify_print_function(
		int *poly, int sz, void *data)
{
	boolean_function_classify *BFC = (boolean_function_classify *) data;
	geometry::geometry_global Gg;
	ring_theory::longinteger_object a;

	BFC->BF->evaluate(poly + 1, BFC->BF->f_proj);
	Int_vec_copy(BFC->BF->f_proj, BFC->BF->f_proj2, BFC->BF->Q);
	Gg.AG_element_rank_longinteger(2, BFC->BF->f_proj2, 1, BFC->BF->Q, a);

	cout << " : ";
	Int_vec_print(cout, BFC->BF->f_proj, BFC->BF->Q);
	cout << " : rk=" << a;

}

static void boolean_function_classify_reduction_function(
		int *poly, void *data)
{
	boolean_function_classify *BFC = (boolean_function_classify *) data;

	if (BFC->BF->dim_kernel) {
		int i, i1, i2;
		int a, ma;

		for (i = 0; i < BFC->BF->dim_kernel; i++) {
			i1 = BFC->BF->Kernel[i * 2 + 0];
			i2 = BFC->BF->Kernel[i * 2 + 1];
			a = poly[i1];
			if (a) {
				ma = BFC->BF->Fq->negate(a);
				poly[i1] = 0;
				poly[i2] = BFC->BF->Fq->add(poly[i2], ma);
			}

		}
	}
#if 0
	// c_0 = c_4:
	a = poly[0];
	if (a) {
		ma = BFC->Fq->negate(a);
		poly[0] = 0;
		poly[4] = BFC->Fq->add(poly[4], ma);
	}
	// c_1 = c_5:
	a = poly[1];
	if (a) {
		ma = BFC->Fq->negate(a);
		poly[1] = 0;
		poly[5] = BFC->Fq->add(poly[5], ma);
	}
#endif
	//BFC->evaluate(poly + 1, BFC->f_proj);
	//int_vec_print(cout, BFC->f_proj, BFC->Q);

}



}}}


