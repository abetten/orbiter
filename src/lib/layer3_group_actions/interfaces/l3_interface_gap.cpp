/*
 * l3_interface_gap.cpp
 *
 *  Created on: Jan 28, 2023
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace interfaces {


l3_interface_gap::l3_interface_gap()
{
	Record_birth();

}


l3_interface_gap::~l3_interface_gap()
{
	Record_death();

}

void l3_interface_gap::canonical_image_GAP(
		groups::strong_generators *SG,
		long int *set, int sz,
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
		SG->A->Group_element->element_print_as_permutation_with_offset(
				SG->gens->ith(i), ost,
				1 /*offset*/,
				true /* f_do_it_anyway_even_for_big_degree */,
				false /* f_print_cycles_of_length_one */,
				0 /* verbose_level*/);
		if (i < SG->gens->len - 1) {
			ost << ", " << endl;
		}
	}
	ost << "]);" << endl;

	other::data_structures::string_tools ST;
	std::string output;

	long int *set2;

	set2 = NEW_lint(sz);

	// add one because GAP permutation domains are 1-based:
	for (i = 0; i < sz; i++) {
		set2[i] = set[i] + 1;
	}

	ST.create_comma_separated_list(output, set2, sz);

	ost << "LoadPackage(\"images\");" << endl;
	ost << "MinimalImage(G, [" << output << "], OnSets);" << endl;

	FREE_lint(set2);

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
	other::l1_interfaces::interface_gap_low Interface_low;
	algebra::field_theory::finite_field *F;
	algebra::basic_algebra::matrix_group *M;
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

		ost << "frob" << h + 1 << " := FrobeniusAutomorphism("
				"GF(" << F->q << "))^" << frob << ";" << endl;

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


void l3_interface_gap::export_surface(
		std::ostream &ost,
		std::string &label_txt,
		int f_has_group,
		groups::strong_generators *SG,
		algebra::ring_theory::homogeneous_polynomial_domain *Poly3_4,
		int *equation,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "l3_interface_gap::export_surface" << endl;
	}

	other::data_structures::string_tools ST;

	ost << "# Cubic surface " << label_txt << endl;
	ost << "# Group:" << endl;

	if (f_has_group) {

		if (!SG->A->is_matrix_group()) {
			cout << "l3_interface_gap::export_surface "
					"the group is not a matrix group" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "l3_interface_gap::export_surface "
					"before Sg->export_fining" << endl;
		}
		SG->export_fining(SG->A, ost, verbose_level);
		if (f_v) {
			cout << "l3_interface_gap::export_surface "
					"after Sg->export_fining" << endl;
		}
	}
	else {
		cout << "l3_interface_gap::export_surface "
				"the group is not available" << endl;
	}

	//SO->Surf->print_equation_with_line_breaks_tex(ost, SO->eqn);

	other::data_structures::string_tools String;
	std::stringstream ss;
	string s;


	//r:=PolynomialRing(GF(x),["X0","X1","X2","X3"]);

	ost << "r := PolynomialRing(GF(" << Poly3_4->get_F()->q << "),"
			"[\"X0\",\"X1\",\"X2\",\"X3\"]);" << endl;

	Poly3_4->print_equation_for_gap_str(
			ss, equation);

	s = ss.str();
	String.remove_specific_character(s, '_');


	ost << "Eqn := " << s << ";" << endl;



	if (f_v) {
		cout << "l3_interface_gap::export_surface done" << endl;
	}

}


void l3_interface_gap::export_BLT_set(
		std::ostream &ost,
		std::string &label_txt,
		int f_has_group,
		groups::strong_generators *SG,
		actions::action *A,
		layer1_foundations::geometry::orthogonal_geometry::blt_set_domain
				*Blt_set_domain,
		long int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "l3_interface_gap::export_BLT_set" << endl;
	}


	ost << "# BLT-set " << label_txt << endl;



	if (f_has_group) {

		algebra::ring_theory::longinteger_object go;

		SG->group_order(go);
		ost << "# Group of order " << go << endl;

		if (!SG->A->is_matrix_group()) {
			cout << "l3_interface_gap::export_BLT_set "
					"the group is not a matrix group" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "l3_interface_gap::export_BLT_set "
					"before SG->export_fining" << endl;
		}
		SG->export_fining(A, ost, verbose_level);
		if (f_v) {
			cout << "l3_interface_gap::export_BLT_set "
					"after SG->export_fining" << endl;
		}
	}
	else {
		cout << "l3_interface_gap::export_BLT_set "
				"the group is not available" << endl;
	}

	int h, i, a;
	int sz;
	int d = 5;
	int v[5];
	other::l1_interfaces::interface_gap_low Interface;

	sz = Blt_set_domain->target_size;

	ost << "pg := ProjectiveSpace(" << 4 << "," << Blt_set_domain->F->q << ");" << endl;
	ost << "S:=[" << endl;
	for (h = 0; h < sz; h++) {

		Blt_set_domain->O->Hyperbolic_pair->unrank_point(v, 1, set[h], 0);


		Blt_set_domain->F->Projective_space_basic->PG_element_normalize_from_front(v, 1, 5);

		ost << "[";
		for (i = 0; i < d; i++) {
			a = v[i];

			Interface.write_element_of_finite_field(ost, Blt_set_domain->F, a);

			if (i < d - 1) {
				ost << ",";
			}
		}
		ost << "]";
		if (h < sz - 1) {
			ost << ",";
		}
		ost << endl;
	}

	ost << "];" << endl;
	ost << "S := List(S,x -> VectorSpaceToElement(pg,x));" << endl;



	if (f_v) {
		cout << "l3_interface_gap::export_BLT_set done" << endl;
	}
}

void l3_interface_gap::export_group_to_GAP_and_copy_to_latex(
		std::ostream &ost,
		std::string &label_txt,
		groups::strong_generators *SG,
		actions::action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "l3_interface_gap::export_group_to_GAP_and_copy_to_latex" << endl;
	}
	string export_fname;

	export_fname = label_txt + "_group.gap";

	export_permutation_group_to_GAP(
			export_fname, A2, SG, verbose_level - 2);

	if (f_v) {
		cout << "written file " << export_fname << " of size "
				<< Fio.file_size(export_fname) << endl;
	}

	ost << "\\subsection*{GAP Export}" << endl;
	ost << "To export the group to GAP, "
			"use the following file\\\\" << endl;
	ost << "\\begin{verbatim}" << endl;

	{
		ifstream fp1(export_fname);
		char line[100000];

		while (true) {
			if (fp1.eof()) {
				break;
			}

			//cout << "count_number_of_orbits_in_file reading
			//line, nb_sol = " << nb_sol << endl;
			fp1.getline(line, 100000, '\n');
			ost << line << endl;
		}

	}
	ost << "\\end{verbatim}" << endl;

	if (f_v) {
		cout << "l3_interface_gap::export_group_to_GAP_and_copy_to_latex done" << endl;
	}
}

void l3_interface_gap::export_permutation_group_to_GAP(
		std::string &fname,
		actions::action *A2,
		groups::strong_generators *SG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "l3_interface_gap::export_permutation_group_to_GAP" << endl;
	}
	{
		ofstream fp(fname);

		fp << "G := Group([" << endl;
		for (i = 0; i < SG->gens->len; i++) {
			A2->Group_element->element_print_as_permutation_with_offset(
				SG->gens->ith(i), fp,
				1 /* offset */,
				true /* f_do_it_anyway_even_for_big_degree */,
				false /* f_print_cycles_of_length_one */,
				0 /* verbose_level */);
			if (i < SG->gens->len - 1) {
				fp << ", " << endl;
			}
		}
		fp << "]);" << endl;

	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "l3_interface_gap::export_permutation_group_to_GAP done" << endl;
	}
}



}}}



