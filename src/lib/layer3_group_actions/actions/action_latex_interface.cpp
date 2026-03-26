/*
 * action_latex_interface.cpp
 *
 *  Created on: Mar 15, 2026
 *      Author: betten
 */






#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


action_latex_interface::action_latex_interface()
{
	Record_birth();

	A = NULL;
}

action_latex_interface::~action_latex_interface()
{
	Record_death();
}

void action_latex_interface::init(
		actions::action *A, int verbose_level)
{
	action_latex_interface::A = A;
}

void action_latex_interface::report(
		std::ostream &ost,
		int f_sims, groups::sims *S,
		int f_strong_gens, groups::strong_generators *SG,
		other::graphics::draw_options *LG_Draw_options,
		int verbose_level)
// reports the sims object from the arguments
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report" << endl;
		cout << "action_latex_interface::report verbose_level = " << verbose_level << endl;
	}

	ost << "\\section*{The Action}" << endl;


	if (f_v) {
		cout << "action_latex_interface::report "
				"before report_group_name_and_degree" << endl;
	}
	report_group_name_and_degree(
			ost,
			verbose_level - 1);
	if (f_v) {
		cout << "action_latex_interface::report "
				"after report_group_name_and_degree" << endl;
	}

	if (f_strong_gens) {

		algebra::ring_theory::longinteger_object go;

		SG->group_order(go);
		ost << "Group order = " << go << "\\\\" << endl;

	}

#if 0
	if (label_tex.length() == 0) {
		cout << "action_latex_interface::report the group has no tex-name" << endl;
		exit(1);
	}
	ost << "Group action $" << label_tex
			<< "$ of degree " << degree << "\\\\" << endl;
#endif


	if (f_v) {
		cout << "action_latex_interface::report before report_what_we_act_on" << endl;
	}
	report_what_we_act_on(
			ost,
			verbose_level - 1);
	if (f_v) {
		cout << "action_latex_interface::report after report_what_we_act_on" << endl;
	}

	if (A->is_matrix_group()) {
		ost << "The group is a matrix group.\\\\" << endl;

#if 0
		field_theory::finite_field *F;
		groups::matrix_group *M;

		M = get_matrix_group();
		F = M->GFq;

		{
			geometry::projective_space *P;

			P = NEW_OBJECT(geometry::projective_space);

			P->projective_space_init(M->n - 1, F, true, verbose_level);

			ost << "The base action is on projective space ${\\rm PG}(" << M->n - 1 << ", " << F->q << ")$\\\\" << endl;

			P->Reporting->report_summary(ost);



			FREE_OBJECT(P);
		}
#endif


	}

	if (A->type_G == wreath_product_t) {
		group_constructions::wreath_product *W;

		W = A->G.wreath_product_group;
		if (f_v) {
			cout << "action_latex_interface::report before W->report" << endl;
		}
		W->report(ost, verbose_level - 1);
		if (f_v) {
			cout << "action_latex_interface::report after W->report" << endl;
		}
	}

	ost << "\\subsection*{Base and Stabilizer Chain}" << endl;

	if (f_sims) {
		if (f_v) {
			cout << "action_latex_interface::report we have sims, printing group order" << endl;
		}
		algebra::ring_theory::longinteger_object go;

		S->group_order(go);
		ost << "Group order " << go << "\\\\" << endl;
		ost << "tl=$";
		//int_vec_print(ost, S->orbit_len, base_len());
		for (int t = 0; t < S->A->base_len(); t++) {
			ost << S->get_orbit_length(t);
			if (t < S->A->base_len()) {
				ost << ", ";
			}
		}
		ost << "$\\\\" << endl;
		if (f_v) {
			cout << "action_latex_interface::report printing group order done" << endl;
		}
	}

	if (A->Stabilizer_chain) {
		if (f_v) {
			cout << "action_latex_interface::report Stabilizer_chain is allocated" << endl;
		}
		if (A->base_len()) {
			ost << "action\\_latex\\_interface::report\\\\" << endl;

			report_base(
					ost,
					verbose_level);

#if 0
			ost << "Base of length " << base_len() << ": $";
			Lint_vec_print(ost, get_base(), base_len());
			ost << "$\\\\" << endl;

			int i;
			ost << "Base = \\\\" << endl;
			for (i = 0; i < base_len(); i++) {
				string s1;

				s1 = Group_element->stringify_point(base_i(i), verbose_level - 1);
				ost << i << " : " << base_i(i) << " = $(" << s1 << ")$\\\\" << endl;
			}
#endif

		}
		if (f_strong_gens) {
			ost << "{\\small\\arraycolsep=2pt" << endl;
			SG->print_generators_tex(ost);
			ost << "}" << endl;
		}
		else {
			ost << "Does not have strong generators.\\\\" << endl;
		}
	}
	if (f_sims) {
		if (f_v) {
			cout << "action_latex_interface::report before S->report" << endl;
		}
		S->report(ost, A->label, LG_Draw_options, verbose_level - 2);
		if (f_v) {
			cout << "action_latex_interface::report after S->report" << endl;
		}
	}
	if (A->Stabilizer_chain && A->base_len() > 0) {
		if (f_v) {
			cout << "action_latex_interface::report we have Stabilizer_chain" << endl;
		}
		if (f_strong_gens) {

			if (f_v) {
				cout << "action_latex_interface::report we have f_strong_gens" << endl;
			}

			action_global Global;

			if (f_v) {
				cout << "action_latex_interface::report "
						"before Global.report_strong_generators" << endl;
			}
			Global.report_strong_generators(
					ost,
					SG,
					A,
					verbose_level);
			if (f_v) {
				cout << "action_latex_interface::report "
						"after Global.report_strong_generators" << endl;
			}


		}
	}
	if (f_v) {
		cout << "action_latex_interface::report done" << endl;
	}
}


void action_latex_interface::report_base(
		std::ostream &ost,
		int verbose_level)
{
	ost << "Base of length " << A->base_len() << ": $";
	Lint_vec_print(ost, A->get_base(), A->base_len());
	ost << "$\\\\" << endl;

	int i;
	ost << "Base = \\\\" << endl;
	for (i = 0; i < A->base_len(); i++) {
		string s1;

		s1 = A->Group_element->stringify_point(A->base_i(i), verbose_level - 1);
		ost << i << " : " << A->base_i(i) << " = $(" << s1 << ")$\\\\" << endl;
	}

}
void action_latex_interface::report_group_name_and_degree(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report_group_name_and_degree" << endl;
	}

	if (A->label_tex.length() == 0) {
		cout << "action_latex_interface::report_group_name_and_degree "
				"the group has no tex-name" << endl;
		exit(1);
	}
	ost << "Group action $" << A->label_tex
			<< "$ of degree " << A->degree << "\\\\" << endl;


}

void action_latex_interface::report_type_of_action(
		std::ostream &ost,
		int verbose_level)
{
	std::string txt;
	std::string tex;
	action_global AcGl;

	AcGl.get_symmetry_group_type_text(txt, tex, A->type_G);


	ost << "The action is of type " << tex << "\\\\" << endl;

	ost << "\\bigskip" << endl;

}

void action_latex_interface::report_what_we_act_on(
		std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report_what_we_act_on" << endl;
	}


	if (f_v) {
		cout << "action_latex_interface::report_what_we_act_on "
				"before report_type_of_action" << endl;
	}
	report_type_of_action(ost, verbose_level);
	if (f_v) {
		cout << "action_latex_interface::report_what_we_act_on "
				"after report_type_of_action" << endl;
	}

	if (A->is_matrix_group()) {

		if (f_v) {
			cout << "action_latex_interface::report_what_we_act_on "
					"is_matrix_group is true" << endl;
		}
		algebra::field_theory::finite_field *F;
		algebra::basic_algebra::matrix_group *M;

		M = A->get_matrix_group();
		F = M->GFq;

#if 0
		{
			geometry::projective_space *P;

			P = NEW_OBJECT(geometry::projective_space);

			P->projective_space_init(M->n - 1, F, true, verbose_level);

			ost << "\\section*{The Group Acts on Projective Space ${\\rm PG}(" << M->n - 1 << ", " << F->q << ")$}" << endl;

			P->Reporting->report(ost, O, verbose_level);



			FREE_OBJECT(P);
		}
#endif

		if (A->type_G == action_on_orthogonal_t) {

			if (A->G.AO->f_on_points) {
				ost << "acting on points only\\\\" << endl;
				ost << "Number of points = "
						<< A->G.AO->O->Hyperbolic_pair->nb_points << "\\\\" << endl;
			}
			else if (A->G.AO->f_on_lines) {
				ost << "acting on lines only\\\\" << endl;
				ost << "Number of lines = "
						<< A->G.AO->O->Hyperbolic_pair->nb_lines << "\\\\" << endl;
			}
			else if (A->G.AO->f_on_points_and_lines) {
				ost << "acting on points and lines\\\\" << endl;
				ost << "Number of points = "
						<< A->G.AO->O->Hyperbolic_pair->nb_points << "\\\\" << endl;
				ost << "Number of lines = "
						<< A->G.AO->O->Hyperbolic_pair->nb_lines << "\\\\" << endl;
			}

			A->G.AO->O->Quadratic_form->report_quadratic_form(
					ost, 0 /* verbose_level */);

			ost << "Tactical decomposition induced by a hyperbolic pair:\\\\" << endl;
			A->G.AO->O->report_schemes_easy(ost);

			A->G.AO->O->report_points(ost, 0 /* verbose_level */);

			A->G.AO->O->report_lines(ost, 0 /* verbose_level */);

		}

		if (M->f_projective) {

			ost << "Group Action $" << A->label_tex
					<< "$ on Projective Space ${\\rm PG}"
							"(" << M->n - 1 << ", " << F->q << ")$\\\\" << endl;

		}
		else if (M->f_affine) {

			ost << "Group Action $" << A->label_tex
					<< "$ on Affine Space ${\\rm AG}"
							"(" << M->n << ", " << F->q << ")$\\\\" << endl;

		}
		else if (M->f_general_linear) {

			ost << "Group Action $" << A->label_tex
					<< "$ on Affine Space ${\\rm AG}"
							"(" << M->n << ", " << F->q << ")$\\\\" << endl;

		}

#if 0
		ost << "The finite field ${\\mathbb F}_{" << F->q << "}$:\\\\" << endl;

		F->Io->cheat_sheet(ost, verbose_level);

		ost << endl << "\\bigskip" << endl << endl;
#endif

	}
	else {
		if (f_v) {
			cout << "action_latex_interface::report_what_we_act_on is_matrix_group is false" << endl;
		}
	}

#if 0
	if (degree < 1000) {
		ost << "The group acts on the following set of size " << degree << ":\\\\" << endl;

		if (ptr->ptr_unrank_point) {
			if (f_v) {
				cout << "action_latex_interface::report_what_we_act_on before latex_all_points" << endl;
			}
			latex_all_points(ost);
			if (f_v) {
				cout << "action_latex_interface::report_what_we_act_on after latex_all_points" << endl;
			}
		}
		else {
			ost << "we don't have an unrank point function\\\\" << endl;
		}
	}
#endif



	if (f_v) {
		cout << "action_latex_interface::report_what_we_act_on done" << endl;
	}
}


void action_latex_interface::list_elements_as_permutations_vertically(
		data_structures_groups::vector_ge *gens,
		ostream &ost)
{
	int i, j, a, len;

	len = gens->len;
	for (j = 0; j < len; j++) {
		ost << " & \\alpha_{" << j << "}";
	}
	ost << "\\\\" << endl;
	for (i = 0; i < A->degree; i++) {
		ost << setw(3) << i;
		for (j = 0; j < len; j++) {
			a = A->Group_element->element_image_of(
					i,
					gens->ith(j),
					0 /* verbose_level */);
			ost << " & " << setw(3) << a;
		}
		ost << "\\\\" << endl;
	}
}

void action_latex_interface::report_basic_orbits(
		std::ostream &ost)
{

	if (A->Stabilizer_chain) {
#if 0
		int i;
		ost << "The base has length " << base_len() << "\\\\" << endl;
		ost << "The basic orbits are: \\\\" << endl;
		for (i = 0; i < base_len(); i++) {
			ost << "Basic orbit " << i << " is orbit of " << base_i(i)
				<< " of length " << transversal_length_i(i) << "\\\\" << endl;
		}
#endif
		A->Stabilizer_chain->report_basic_orbits(
				ost);

	}
	else {
		cout << "action " << A->label << " does not have a base" << endl;
	}
}

void action_latex_interface::latex_all_points(
		std::ostream &ost)
{
	int i;
	int *v;


	if (A->ptr->ptr_unrank_point == NULL) {
		cout << "action_latex_interface::latex_all_points ptr->ptr_unrank_point == NULL" << endl;
		return;
	}
	v = NEW_int(A->low_level_point_size);
#if 0
	cout << "action_latex_interface::latex_all_points "
			"low_level_point_size=" << low_level_point_size <<  endl;
	ost << "{\\renewcommand*{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & P_{i}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < degree; i++) {
		unrank_point(i, v);
		ost << i << " & ";
		int_vec_print(ost, v, low_level_point_size);
		ost << "\\\\" << endl;
		if (((i + 1) % 10) == 0) {
			ost << "\\hline" << endl;
			ost << "\\end{array}" << endl;
			if (((i + 1) % 50) == 0) {
				ost << "$$" << endl;
				ost << "$$" << endl;
			}
			else {
				ost << ", \\;" << endl;
			}
			ost << "\\begin{array}{|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "i & P_{i}\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}%" << endl;
	cout << "action_latex_interface::latex_all_points done" << endl;
#else
	if (A->low_level_point_size < 10) {
		ost << "\\begin{multicols}{2}" << endl;
	}
	ost << "\\noindent" << endl;
	for (i = 0; i < A->degree; i++) {
		A->Group_element->unrank_point(i, v);
		ost << i << " = ";
		Int_vec_print(ost, v, A->low_level_point_size);
		ost << "\\\\" << endl;
	}
	if (A->low_level_point_size < 10) {
		ost << "\\end{multicols}" << endl;
	}

#endif

	FREE_int(v);
}

void action_latex_interface::latex_point_set(
		std::ostream &ost,
		long int *set, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *v;

	if (f_v) {
		cout << "action_latex_interface::print_points "
				"low_level_point_size=" << A->low_level_point_size <<  endl;
	}
	v = NEW_int(A->low_level_point_size);
#if 0
	ost << "{\\renewcommand*{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & P_{i} \\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < sz; i++) {
		unrank_point(set[i], v);
		ost << i << " & ";
		ost << set[i] << " = ";
		int_vec_print(ost, v, low_level_point_size);
		ost << "\\\\" << endl;
		if (((i + 1) % 10) == 0) {
			ost << "\\hline" << endl;
			ost << "\\end{array}" << endl;
			if (((i + 1) % 50) == 0) {
				ost << "$$" << endl;
				ost << "$$" << endl;
			}
			else {
				ost << ", \\;" << endl;
			}
			ost << "\\begin{array}{|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "i & P_{i}\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
		}
	}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}%" << endl;
#else


	if (A->ptr->ptr_unrank_point) {
		if (A->low_level_point_size < 10) {
			ost << "\\begin{multicols}{2}" << endl;
		}
		ost << "\\noindent" << endl;
		for (i = 0; i < sz; i++) {
			A->Group_element->unrank_point(set[i], v);
			ost << i << " : ";
			ost << set[i] << " = ";
			Int_vec_print(ost, v, A->low_level_point_size);
			ost << "\\\\" << endl;
		}
		if (A->low_level_point_size < 10) {
			ost << "\\end{multicols}" << endl;
		}
	}
#endif

	FREE_int(v);
	if (f_v) {
		cout << "action_latex_interface::print_points done" << endl;
	}
}

void action_latex_interface::write_set_of_elements_latex_file(
		std::string &fname,
		std::string &title, int *Elt, int nb_elts)
{
	{
		ofstream ost(fname);
		algebra::number_theory::number_theory_domain NT;

		string author, extra_praeamble;

		other::l1_interfaces::latex_interface L;

		L.head(ost,
				false /* f_book*/,
				true /* f_title */,
				title, author,
				false /* f_toc */,
				false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		int i;

		for (i = 0; i < nb_elts; i++) {
			ost << "$$" << endl;
			A->Group_element->element_print_latex(Elt + i * A->elt_size_in_int, ost);
			ost << "$$" << endl;
		}

		L.foot(ost);


	}

	other::orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

}


}}}


