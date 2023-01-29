/*
 * l3_interface_gap.cpp
 *
 *  Created on: Jan 28, 2023
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace interfaces {


l3_interface_gap::l3_interface_gap()
{

}


l3_interface_gap::~l3_interface_gap()
{

}

void l3_interface_gap::canonical_image_GAP(
		groups::strong_generators *SG,
		std::string &input_set_text,
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "l3_interface_gap::canonical_image_GAP" << endl;
	}
	int i;

	//ost << "Generators in GAP format are:" << endl;
	ost << "G := Group([";
	for (i = 0; i < SG->gens->len; i++) {
		SG->A->element_print_as_permutation_with_offset(
				SG->gens->ith(i), ost,
				1 /*offset*/,
				TRUE /* f_do_it_anyway_even_for_big_degree */,
				FALSE /* f_print_cycles_of_length_one */,
				0 /* verbose_level*/);
		if (i < SG->gens->len - 1) {
			ost << ", " << endl;
		}
	}
	ost << "]);" << endl;

	long int *set;
	int sz;
	data_structures::string_tools ST;
	std::string output;


	Get_lint_vector_from_label(input_set_text, set, sz, 0 /* verbose_level */);

	// add one because GAP permutation domains are 1-based:
	for (i = 0; i < sz; i++) {
		set[i]++;
	}

	ST.create_comma_separated_list(output, set, sz);

	ost << "LoadPackage(\"images\");" << endl;
	ost << "MinimalImage(G, [" << output << "], OnSets);" << endl;
	if (f_v) {
		cout << "l3_interface_gap::canonical_image_GAP done" << endl;
	}
}

void l3_interface_gap::export_collineation_group_to_fining(
		std::ostream &ost,
		groups::strong_generators *SG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "l3_interface_gap::export_collineation_group_to_fining" << endl;
	}
	int h;
	algebra::interface_gap_low Interface_low;
	field_theory::finite_field *F;
	groups::matrix_group *M;
	int *Elt;
	int frob;
	int d;


	M = SG->A->get_matrix_group();
	F = M->GFq;
	d = M->n;

	for (h = 0; h < SG->gens->len; h++) {

		Elt = SG->gens->ith(h);

		ost << "mat" << h + 1 << " := ";

		Interface_low.write_matrix(
				ost,
				F,
				Elt, d,
				verbose_level - 1);
		ost << ";" << endl;

		if (M->f_semilinear) {
			frob = Elt[d * d];
		}
		else {
			frob = 0;
		}

		ost << "frob" << h + 1 << " := FrobeniusAutomorphism(GF(" << F->q << "))^" << frob << ";" << endl;

		ost << "psi" << h + 1 << " := ProjectiveSemilinearMap("
				"mat" << h + 1 << ", frob" << h + 1 << ",GF(" << F->q << "));" << endl;
	}
	ost << "gens := [";
	for (h = 0; h < SG->gens->len; h++) {
		ost << "psi" << h + 1;
		if (h < SG->gens->len - 1) {
			ost << ", ";
		}
	}
	ost << "];" << endl;
	ost << "G := Group(gens);" << endl;
	ost << "Size(G);" << endl;

	if (f_v) {
		cout << "l3_interface_gap::export_collineation_group_to_fining done" << endl;
	}
}



}}}



