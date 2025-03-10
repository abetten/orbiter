/*
 * cyclotomic_sets.cpp
 *
 *  Created on: Sep 30, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace number_theory {


cyclotomic_sets::cyclotomic_sets()
{
	Record_birth();
	n = 0;
	q = 0;
	m = 0;
	qm = 0;
	Index = NULL;
	S = NULL;
}

cyclotomic_sets::~cyclotomic_sets()
{
	Record_death();
	if (Index) {
		FREE_int(Index);
	}
	if (S) {
		FREE_OBJECT(S);
	}
}

void cyclotomic_sets::init(
		algebra::field_theory::finite_field *F,
		int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cyclotomic_sets::init, n=" << n << endl;
	}
	cyclotomic_sets::n = n;
	cyclotomic_sets::q = F->q;
	//longinteger_object Qm1, Idx;
	//longinteger_domain D;
	number_theory_domain NT;

	m = NT.order_mod_p(q, n);
	if (f_v) {
		cout << "cyclotomic_sets::init order of q mod n is m=" << m << endl;
	}

	qm = NT.i_power_j(q, m);
	if (f_v) {
		cout << "cyclotomic_sets::init qm = " << qm << endl;
	}

	//D.create_qnm1(Qm1, q, m);

	//D.integral_division_by_int(Qm1, n, Idx, r);

	int i;
	int *f_taken;
	int *transversal;
	long int *set;
	int sz;
	int tl;

	Index = NEW_int(n);
	f_taken = NEW_int(n);
	transversal = NEW_int(n);
	for (i = 0; i < n; i++) {
		f_taken[i] = false;
	}


	S = NEW_OBJECT(other::data_structures::set_of_sets);

	if (f_v) {
		cout << "cyclotomic_sets::init before S->init_simple" << endl;
	}

	S->init_simple(n /* underlying_set_size */,
			n /* nb_sets */, 0 /* verbose_level */);

	if (f_v) {
		cout << "cyclotomic_sets::init after S->init_simple" << endl;
	}

	tl = 0;

	for (i = 0; i < n; i++) {
		if (f_taken[i]) {
			cout << q << "-cyclotomic coset of "
					<< i << " already taken" << endl;
			continue;
		}
		transversal[tl] = i;
		f_taken[i] = true;
		set = NEW_lint(n);
		set[0] = i;
		sz = 1;
		Index[i] = tl;
		if (f_v) {
			cout << q << "-cyclotomic coset of "
					<< i << " : " << i;
		}
		while (true) {
			i = (q * i) % n;
			if (f_taken[i]) {
				break;
			}
			f_taken[i] = true;
			set[sz++] = i;
			Index[i] = tl;
		}
		S->Sets[tl] = set;
		S->Set_size[tl] = sz;
		tl++;
		if (f_v) {
			cout << " has size = " << sz << endl;
		}
	}
	S->nb_sets = tl;

	if (f_v) {
		cout << "cyclotomic_sets::init cyclotomic sets are:" << endl;
		S->print_table();
		cout << "cyclotomic_sets::init Index:" << endl;
		Int_vec_print(cout, Index, n);
		cout << endl;
	}



	if (f_v) {
		cout << "cyclotomic_sets::init done" << endl;
	}
}

void cyclotomic_sets::print()
{
	S->print_table();
}

void cyclotomic_sets::print_latex(
		std::ostream &ost)
{
	S->print_table_latex_simple(ost);
}

void cyclotomic_sets::print_latex_with_selection(
		std::ostream &ost, int *Selection, int nb_sel)
{
	S->print_table_latex_simple_with_selection(ost, Selection, nb_sel);
}


}}}}




