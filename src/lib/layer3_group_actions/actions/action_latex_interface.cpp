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

	ost << "\\section*{The Group and its Action}" << endl;


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
			//ost << "action\\_latex\\_interface::report\\\\" << endl;

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

			//action_global Global;

			if (f_v) {
				cout << "action_latex_interface::report "
						"before report_strong_generators" << endl;
			}
			report_strong_generators(
					ost,
					SG,
					A,
					verbose_level);
			if (f_v) {
				cout << "action_latex_interface::report "
						"after report_strong_generators" << endl;
			}


		}
	}

	ost << "\\bigskip" << endl;


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


void action_latex_interface::report_strong_generators(
		std::ostream &ost,
		groups::strong_generators *SG,
		action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report_strong_generators" << endl;
	}

	// GAP:

	report_strong_generators_GAP(
			ost,
			SG,
			A,
			verbose_level - 1);


	// Fining:

	report_strong_generators_fining(
			ost,
			SG,
			A,
			verbose_level - 1);


	// Magma:

	report_strong_generators_magma(
			ost,
			SG,
			A,
			verbose_level - 1);


	// Orbiter compact form:

	report_strong_generators_orbiter(
			ost,
			SG,
			A,
			verbose_level - 1);




	if (f_v) {
		cout << "action_latex_interface::report_strong_generators done" << endl;
	}
}

void action_latex_interface::report_strong_generators_GAP(
		std::ostream &ost,
		groups::strong_generators *SG,
		action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report_strong_generators_GAP" << endl;
	}


	ost << endl;
	ost << "\\bigskip" << endl;
	ost << endl;

	ost << "GAP export: \\\\" << endl;
	ost << "\\begin{verbatim}" << endl;
	if (f_v) {
		cout << "action_global::report_strong_generators "
				"before SG->print_generators_gap" << endl;
	}
	SG->print_generators_gap(ost, verbose_level - 1);
	if (f_v) {
		cout << "action_global::report_strong_generators "
				"after SG->print_generators_gap" << endl;
	}
	ost << "\\end{verbatim}" << endl;
	if (f_v) {
		cout << "action_latex_interface::report_strong_generators_GAP done" << endl;
	}
}


void action_latex_interface::report_strong_generators_fining(
		std::ostream &ost,
		groups::strong_generators *SG,
		action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report_strong_generators_fining" << endl;
	}

	ost << "Fining export: \\\\" << endl;
	ost << "\\begin{verbatim}" << endl;
	if (f_v) {
		cout << "action_latex_interface::report_strong_generators "
				"before SG->export_fining" << endl;
	}
	SG->export_fining(A, ost, verbose_level);
	if (f_v) {
		cout << "action_latex_interface::report_strong_generators "
				"after SG->export_fining" << endl;
	}
	ost << "\\end{verbatim}" << endl;

	if (f_v) {
		cout << "action_latex_interface::report_strong_generators_fining done" << endl;
	}

}


void action_latex_interface::report_strong_generators_magma(
		std::ostream &ost,
		groups::strong_generators *SG,
		action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report_strong_generators_magma" << endl;
	}

	ost << "Magma export: \\\\" << endl;
	ost << "\\begin{verbatim}" << endl;
	if (f_v) {
		cout << "action_latex_interface::report_strong_generators "
				"before SG->export_magma" << endl;
	}
	SG->export_magma(A, ost, verbose_level);
	if (f_v) {
		cout << "action_latex_interface::report_strong_generators "
				"after SG->export_magma" << endl;
	}
	ost << "\\end{verbatim}" << endl;

	if (f_v) {
		cout << "action_latex_interface::report_strong_generators_magma done" << endl;
	}
}


void action_latex_interface::report_strong_generators_orbiter(
		std::ostream &ost,
		groups::strong_generators *SG,
		action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report_strong_generators_orbiter" << endl;
	}

	ost << "Compact form: \\\\" << endl;
	ost << "\\begin{verbatim}" << endl;
	if (f_v) {
		cout << "action_latex_interface::report_strong_generators "
				"before SG->print_generators_compact" << endl;
	}
	SG->print_generators_compact(ost, verbose_level - 1);
	if (f_v) {
		cout << "action_latex_interface::report_strong_generators "
				"after SG->print_generators_compact" << endl;
	}
	ost << "\\end{verbatim}" << endl;

	if (f_v) {
		cout << "action_latex_interface::report_strong_generators_orbiter done" << endl;
	}
}


void action_latex_interface::report(
		std::ostream &ost,
		std::string &label,
		std::string &label_tex,
		actions::action *A,
		groups::strong_generators *Strong_gens,
		groups::sims *Sims,
		other::graphics::draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//actions::action *A;

	//A = A2;
	if (f_v) {
		cout << "action_latex_interface::report" << endl;
	}

	//groups::sims *H;

	if (f_v) {
		cout << "action_latex_interface::report "
				"creating report for group " << label << endl;
	}

#if 0
	//G = initial_strong_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "action_latex_interface::report "
				"before Strong_gens->create_sims" << endl;
	}
	H = Strong_gens->create_sims(0 /*verbose_level*/);
	if (f_v) {
		cout << "action_latex_interface::report "
				"after Strong_gens->create_sims" << endl;
	}
#endif

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << Sims->group_order_lint() << endl;

	int *Elt;
	algebra::ring_theory::longinteger_object go;

	Elt = NEW_int(A->elt_size_in_int);
	Sims->group_order(go);


	{

		//H->print_all_group_elements_tex(fp);

		algebra::ring_theory::longinteger_object go;
		//sims *G;
		//sims *H;

		//G = initial_strong_gens->create_sims(verbose_level);
		//H = Strong_gens->create_sims(verbose_level);



		ost << "\\section*{The Group $" << label_tex << "$}" << endl;


		Sims->group_order(go);

		ost << "\\noindent The order of the group $"
				<< label_tex
				<< "$ is " << go << "\\\\" << endl;


#if 0
		void stabilizer_chain_base_data::report_basic_orbits(
				std::ostream &ost);
#endif

#if 0
		fp << "\\noindent The field ${\\mathbb F}_{"
				<< F->q
				<< "}$ :\\\\" << endl;
		if (f_v) {
			cout << "action_latex_interface::report before F->cheat_sheet" << endl;
		}
		F->cheat_sheet(fp, verbose_level);
		if (f_v) {
			cout << "action_latex_interface::report after F->cheat_sheet" << endl;
		}
#endif


#if 0
		ost << "\\noindent The group acts on a set of size "
				<< A->degree << "\\\\" << endl;
#endif

		if (f_v) {
			cout << "action_latex_interface::report "
					"before A->report_what_we_act_on" << endl;
		}

		A->Action_latex_interface->report_what_we_act_on(
				ost,
				verbose_level - 2);

		if (f_v) {
			cout << "action_latex_interface::report "
					"after A->report_what_we_act_on" << endl;
		}


#if 0
		if (A->degree < 1000) {

			A->print_points(fp);
		}
#endif

		//cout << "Order H = " << H->group_order_int() << "\\\\" << endl;

#if 0
		if (f_has_nice_gens) {
			ost << "Nice generators:\\\\" << endl;
			nice_gens->print_tex(ost);
		}
		else {
		}
#endif

		cout << "Strong generators:\\\\" << endl;
		ost << "\\section*{Strong generators}" << endl;
		if (f_v) {
			cout << "action_latex_interface::report "
					"before Strong_gens->print_generators_tex" << endl;
		}
		Strong_gens->print_generators_tex(ost);
		if (f_v) {
			cout << "action_latex_interface::report "
					"after Strong_gens->print_generators_tex" << endl;
		}

		if (A != Strong_gens->A) {

			ost << "\\section*{Strong generators in the induced action}" << endl;
			ost << "Strong generators in the induced action:\\\\" << endl;
			Strong_gens->print_generators_in_different_action_tex(
					ost, A);
		}


		if (f_v) {
			cout << "action_latex_interface::report "
					"before A->Action_latex_interface->report" << endl;
		}

		A->Action_latex_interface->report(
				ost, true /*f_sims*/, Sims,
				true /* f_strong_gens */, Strong_gens,
				LG_Draw_options,
				verbose_level - 2);

		if (f_v) {
			cout << "action_latex_interface::report "
					"after A->Action_latex_interface->report" << endl;
		}

		if (f_v) {
			cout << "action_latex_interface::report before A->report_basic_orbits" << endl;
		}

		A->Action_latex_interface->report_basic_orbits(ost);

		if (f_v) {
			cout << "action_latex_interface::report after A->report_basic_orbits" << endl;
		}



#if 0
		if (f_conjugacy_classes_and_normalizers) {


			interfaces::magma_interface M;


			if (f_v) {
				cout << "action_latex_interface::report f_conjugacy_classes_and_normalizers is true" << endl;
			}

			M.report_conjugacy_classes_and_normalizers(A2, ost, H,
					verbose_level);

			if (f_v) {
				cout << "action_latex_interface::report A2->report_conjugacy_classes_and_normalizers" << endl;
			}
		}
#endif


		//L.foot(fp);
	}

	//FREE_OBJECT(H)
	FREE_int(Elt);
	if (f_v) {
		cout << "action_latex_interface::report creating report for group " << label << " done" << endl;
	}

}


void action_latex_interface::report_order_invariant(
		std::ostream &ost,
		std::string &label,
		std::string &label_tex,
		actions::action *A,
		groups::strong_generators *Strong_gens,
		groups::sims *Sims,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report_order_invariant" << endl;
	}


	if (f_v) {
		cout << "action_latex_interface::report_order_invariant "
				"creating report on the order invariant for group " << label << endl;
	}


	{


		algebra::ring_theory::longinteger_object go;


		ost << "\\section*{The Group $" << label_tex << "$}" << endl;


		Sims->group_order(go);

		ost << "\\noindent The order of the group $"
				<< label_tex
				<< "$ is " << go << "\\\\" << endl;


		groups::group_theory_global Group_theory_global;
		std::string s;

			s = Group_theory_global.order_invariant(
					A, Strong_gens,
					verbose_level - 3);

		ost << "The order invariant is ";
		ost << "$" << s << "$";
		ost << "\\\\" << endl;

	}

	if (f_v) {
		cout << "action_latex_interface::report_order_invariant done" << endl;
	}
}

void action_latex_interface::report_group_table(
		std::ostream &ost,
		std::string &label,
		std::string &label_tex,
		actions::action *A,
		groups::strong_generators *Strong_gens,
		other::graphics::draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report_group_table" << endl;
	}


	groups::sims *H;

	if (f_v) {
		cout << "action_latex_interface::report_group_table "
				"creating report for group " << label << endl;
	}

	//G = initial_strong_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "action_latex_interface::report_group_table "
				"before Strong_gens->create_sims" << endl;
	}
	H = Strong_gens->create_sims(0 /*verbose_level*/);
	if (f_v) {
		cout << "action_latex_interface::report_group_table "
				"after Strong_gens->create_sims" << endl;
	}

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;


	int *Table;
	long int n;
	other::orbiter_kernel_system::file_io Fio;
	string fname_group_table;
	H->create_group_table(Table, n, verbose_level);

	cout << "action_latex_interface::report_group_table The group table is:" << endl;
	Int_matrix_print(Table, n, n);

	fname_group_table = A->label + "_group_table.csv";
	Fio.Csv_file_support->int_matrix_write_csv(
			fname_group_table, Table, n, n);
	cout << "Written file " << fname_group_table << " of size "
			<< Fio.file_size(fname_group_table) << endl;

	{
		other::l1_interfaces::latex_interface L;

		ost << "\\begin{sidewaystable}" << endl;
		ost << "$$" << endl;
		L.int_matrix_print_tex(ost, Table, n, n);
		ost << "$$" << endl;
		ost << "\\end{sidewaystable}" << endl;

		//int f_with_permutation = false;
		int f_override_action = false;
		actions::action *A_special = NULL;

		H->print_all_group_elements_tex(ost,
				//f_with_permutation,
				f_override_action, A_special);

	}

	{
		string fname2;
		//int x_min = 0, y_min = 0;
		//int xmax = ONE_MILLION;
		//int ymax = ONE_MILLION;

		//int f_embedded = true;
		//int f_sideways = false;
		int *labels;

		int i;

		labels = NEW_int(2 * n);

		for (i = 0; i < n; i++) {
			labels[i] = i;
		}
		if (n > 100) {
			for (i = 0; i < n; i++) {
				labels[n + i] = n + i % 100;
			}
		}
		else {
			for (i = 0; i < n; i++) {
				labels[n + i] = n + i;
			}
		}

		fname2 = A->label + "_group_table_order_" + std::to_string(n);

		{
			other::graphics::mp_graphics G;

			G.init(fname2, LG_Draw_options, verbose_level);

#if 0
			mp_graphics G(fname2, x_min, y_min, xmax, ymax, f_embedded, f_sideways, verbose_level - 1);
			//G.setup(fname2, 0, 0, ONE_MILLION, ONE_MILLION, xmax, ymax, f_embedded, scale, line_width);
			G.out_xmin() = 0;
			G.out_ymin() = 0;
			G.out_xmax() = xmax;
			G.out_ymax() = ymax;
			//cout << "xmax/ymax = " << xmax << " / " << ymax << endl;

			//G.tikz_global_scale = LG_Draw_options->scale;
			//G.tikz_global_line_width = LG_Draw_options->line_width;
#endif

			G.header();
			G.begin_figure(1000 /* factor_1000*/);

			int color_scale[] = {8,5,6,4,3,2,18,19, 7,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,1};
			int nb_colors = sizeof(color_scale) / sizeof(int);

			G.draw_matrix_in_color(
				false /* f_row_grid */, false /* f_col_grid */,
				Table  /* Table */, n /* nb_colors */,
				n, n, //xmax, ymax,
				color_scale, nb_colors,
				true /* f_has_labels */, labels);

			G.finish(cout, true);
		}
		FREE_int(labels);

	}


	FREE_int(Table);
	FREE_OBJECT(H)


	if (f_v) {
		cout << "action_latex_interface::report_group_table done" << endl;
	}

}



void action_latex_interface::report_sylow(
		std::ostream &ost,
		std::string &label,
		std::string &label_tex,
		actions::action *A,
		groups::strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_latex_interface::report_sylow" << endl;
	}


	groups::sims *H;

	if (f_v) {
		cout << "action_latex_interface::report_sylow "
				"creating report for group " << label << endl;
	}

	//G = initial_strong_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "action_latex_interface::report_sylow "
				"before Strong_gens->create_sims" << endl;
	}
	H = Strong_gens->create_sims(0 /*verbose_level*/);
	if (f_v) {
		cout << "action_latex_interface::report_sylow "
				"after Strong_gens->create_sims" << endl;
	}

	//cout << "group order G = " << G->group_order_int() << endl;
	cout << "group order H = " << H->group_order_lint() << endl;




	groups::sylow_structure *Syl;

	Syl = NEW_OBJECT(groups::sylow_structure);
	if (f_v) {
		cout << "action_latex_interface::report_sylow before Syl->init" << endl;
	}
	Syl->init(
			H,
			label,
			label_tex,
			verbose_level - 2);
	if (f_v) {
		cout << "action_latex_interface::report_sylow after Syl->init" << endl;
	}
	if (f_v) {
		cout << "action_latex_interface::report_sylow before Syl->report" << endl;
	}
	Syl->report(ost);
	if (f_v) {
		cout << "action_latex_interface::report_sylow after Syl->report" << endl;
	}



	FREE_OBJECT(Syl)
	FREE_OBJECT(H)

	if (f_v) {
		cout << "action_latex_interface::report_sylow done" << endl;
	}

}


void action_latex_interface::report_groups_and_normalizers(
		action *A,
		std::ostream &ost,
		int nb_subgroups,
		groups::strong_generators *H_gens,
		groups::strong_generators *N_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int u;
	algebra::ring_theory::longinteger_object go1, go2;

	if (f_v) {
		cout << "action_latex_interface::report_groups_and_normalizers" << endl;
	}

	for (u = 0; u < nb_subgroups; u++) {

		ost << "\\subsection*{Class " << u << " / " << nb_subgroups << "}" << endl;

		H_gens[u].group_order(go1);
		N_gens[u].group_order(go2);

		ost << "Group order = " << go1 << "\\\\" << endl;
		ost << "Normalizer order = " << go2 << "\\\\" << endl;

		ost << "Generators for $H$:\\\\" << endl;

		H_gens[u].print_generators_in_latex_individually(ost, verbose_level - 1);
		H_gens[u].print_generators_as_permutations_tex(ost, A);

		ost << "\\bigskip" << endl;

		ost << "Generators for $N(H)$:\\\\" << endl;

		N_gens[u].print_generators_in_latex_individually(ost, verbose_level - 1);
		N_gens[u].print_generators_as_permutations_tex(ost, A);

	}


	if (f_v) {
		cout << "action_latex_interface::report_groups_and_normalizers done" << endl;
	}
}




}}}


