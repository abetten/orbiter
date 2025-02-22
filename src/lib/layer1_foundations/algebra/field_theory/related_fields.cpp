/*
 * related_fields.cpp
 *
 *  Created on: Jan 22, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace field_theory {




related_fields::related_fields()
{
	Record_birth();
	F = NULL;
	nb_subfields = 0;
	Subfield_order = NULL;
	Subfield_exponent = NULL;
	Subfield_index = NULL;
	Subfield_minimum_polynomial = NULL;
	Subfield = NULL;
	SubS = NULL;
}

related_fields::~related_fields()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "related_fields::~related_fields" << endl;
	}
	Record_death();
	if (Subfield_order) {
		if (f_v) {
			cout << "related_fields::~related_fields before FREE_int(Subfield_order)" << endl;
		}
		FREE_int(Subfield_order);
		if (f_v) {
			cout << "related_fields::~related_fields after FREE_int(Subfield_order)" << endl;
		}
	}
	if (Subfield_exponent) {
		if (f_v) {
			cout << "related_fields::~related_fields before FREE_int(Subfield_exponent)" << endl;
		}
		FREE_int(Subfield_exponent);
		if (f_v) {
			cout << "related_fields::~related_fields after FREE_int(Subfield_exponent)" << endl;
		}
	}
	if (Subfield_index) {
		if (f_v) {
			cout << "related_fields::~related_fields before FREE_int(Subfield_index)" << endl;
		}
		FREE_int(Subfield_index);
		if (f_v) {
			cout << "related_fields::~related_fields after FREE_int(Subfield_index)" << endl;
		}
	}
	if (Subfield_minimum_polynomial) {
		if (f_v) {
			cout << "related_fields::~related_fields before FREE_OBJECTS(Subfield_minimum_polynomial)" << endl;
		}
		FREE_OBJECTS(Subfield_minimum_polynomial);
		if (f_v) {
			cout << "related_fields::~related_fields after FREE_OBJECTS(Subfield_minimum_polynomial)" << endl;
		}
	}
	if (Subfield) {
		if (f_v) {
			cout << "related_fields::~related_fields before FREE_OBJECTS(Subfield)" << endl;
		}
		FREE_OBJECTS(Subfield);
		if (f_v) {
			cout << "related_fields::~related_fields after FREE_OBJECTS(Subfield)" << endl;
		}
	}
	if (SubS) {
		if (f_v) {
			cout << "related_fields::~related_fields before FREE_OBJECTS(SubS)" << endl;
		}
		FREE_OBJECTS(SubS);
		if (f_v) {
			cout << "related_fields::~related_fields after FREE_OBJECTS(SubS)" << endl;
		}
	}
	if (f_v) {
		cout << "related_fields::~related_fields done" << endl;
	}
}


void related_fields::init(
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "related_fields::init" << endl;
	}

	related_fields::F = F;
	int h1, h2;
	number_theory::number_theory_domain NT;

	nb_subfields = 0;
	for (h1 = 1; h1 < F->e; h1++) {
		if ((F->e % h1) == 0) {
			nb_subfields++;
		}
	}

	if (f_v) {
		cout << "related_fields::init nb_subfields = " << nb_subfields << endl;
	}


	Subfield_order = NEW_int(nb_subfields);
	Subfield_exponent = NEW_int(nb_subfields);
	Subfield_index = NEW_int(nb_subfields);

	h2 = 0;
	for (h1 = 1; h1 < F->e; h1++) {
		if ((F->e % h1) == 0) {
			Subfield_order[h2] = NT.i_power_j(F->p, h1);
			Subfield_exponent[h2] = h1;
			Subfield_index[h2] = F->e / h1;
			h2++;
		}
	}

	Subfield_minimum_polynomial = NEW_OBJECTS(minimum_polynomial, nb_subfields);

	for (h1 = 0; h1 < nb_subfields; h1++) {

		int order_subfield;


		if (Subfield_exponent[h1] > 1) {
			if (f_v) {
				cout << "related_fields::init "
						"computing minimum polynomial of subfield of order " << Subfield_order[h1] << endl;
			}
			order_subfield = Subfield_order[h1];
			Subfield_minimum_polynomial[h1].compute_subfield_polynomial(
				F,
				order_subfield, verbose_level);
		}
		else {
			if (f_v) {
				cout << "related_fields::init "
						"skipping the prime field" << endl;
			}
		}
	}

	if (f_v) {
		print(cout);
	}

	Subfield = NEW_OBJECTS(finite_field, nb_subfields);

	for (h1 = 0; h1 < nb_subfields; h1++) {

		if (f_v) {
			cout << "related_fields::init "
					"setting up subfield of order " << Subfield_order[h1]
					<< " using the polynomial "
					<< Subfield_minimum_polynomial[h1].min_poly_rank_as_string << endl;
		}
		Subfield[h1].init_override_polynomial_small_order(
				Subfield_order[h1],
				Subfield_minimum_polynomial[h1].min_poly_rank_as_string,
				false /* f_without_tables */,
				false /* f_compute_related_fields */,
				verbose_level);
	}


	SubS = NEW_OBJECTS(subfield_structure, nb_subfields);

	for (h1 = 0; h1 < nb_subfields; h1++) {

		if (f_v) {
			cout << "related_fields::init "
					"setting up subfield structure of order " << Subfield_order[h1] << endl;
		}
		SubS[h1].init(F, &Subfield[h1], verbose_level);

	}

	if (f_v) {
		cout << "related_fields::init done" << endl;
	}
}

void related_fields::print(
		std::ostream &ost)
{
	ost << "Number of (true) subfields: " << nb_subfields << endl;
	int i;
	cout << "i : Subfield_order[i] : Subfield_exponent[i] : Subfield_index[i] : minimum_poly_rank[i]" << endl;
	for (i = 0; i < nb_subfields; i++) {
		ost << i << " : " << Subfield_order[i] << " : " << Subfield_exponent[i] << " : " << Subfield_index[i] << " : " << Subfield_minimum_polynomial[i].min_poly_rank << endl;
	}
}

int related_fields::position_of_subfield(
		int order_of_subfield)
{
	number_theory::number_theory_domain NT;
	other::data_structures::sorting Sorting;

	int p;
	int e;
	int idx;

	NT.factor_prime_power(order_of_subfield, p, e);
	if (p != F->p) {
		cout << "related_fields::position_of_subfield "
				"the given order is not the order of a subfield" << endl;
		exit(1);
	}
	if (!Sorting.int_vec_search(Subfield_exponent, nb_subfields, e, idx)) {
		cout << "related_fields::position_of_subfield "
				"the given order cannot be found." << endl;
		exit(1);

	}
	return idx;
}


}}}}

