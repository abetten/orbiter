/*
 * action_io.cpp
 *
 *  Created on: Mar 30, 2019
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


void action::report(
		std::ostream &ost,
		int f_sims, groups::sims *S,
		int f_strong_gens, groups::strong_generators *SG,
		graphics::layered_graph_draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::report" << endl;
	}

	ost << "\\section*{The Action}" << endl;


	report_group_name_and_degree(
			ost,
			LG_Draw_options,
			verbose_level);

	if (f_strong_gens) {

		ring_theory::longinteger_object go;

		SG->group_order(go);
		ost << "Group order = " << go << "\\\\" << endl;

	}

#if 0
	if (label_tex.length() == 0) {
		cout << "action::report the group has no tex-name" << endl;
		exit(1);
	}
	ost << "Group action $" << label_tex
			<< "$ of degree " << degree << "\\\\" << endl;
#endif


	if (f_v) {
		cout << "action::report before report_what_we_act_on" << endl;
	}
	report_what_we_act_on(
			ost,
			LG_Draw_options,
			verbose_level);
	if (f_v) {
		cout << "action::report after report_what_we_act_on" << endl;
	}

	if (is_matrix_group()) {
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

	if (type_G == wreath_product_t) {
		group_constructions::wreath_product *W;

		W = G.wreath_product_group;
		if (f_v) {
			cout << "action::report before W->report" << endl;
		}
		W->report(ost, verbose_level);
		if (f_v) {
			cout << "action::report after W->report" << endl;
		}
	}

	ost << "\\subsection*{Base and Stabilizer Chain}" << endl;

	if (f_sims) {
		if (f_v) {
			cout << "action::report we have sims, printing group order" << endl;
		}
		ring_theory::longinteger_object go;

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
			cout << "action::report printing group order done" << endl;
		}
	}

	if (Stabilizer_chain) {
		if (base_len()) {
			ost << "Base: $";
			Lint_vec_print(ost, get_base(), base_len());
			ost << "$\\\\" << endl;
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
			cout << "action::report before S->report" << endl;
		}
		S->report(ost, label, LG_Draw_options, verbose_level - 2);
		if (f_v) {
			cout << "action::report after S->report" << endl;
		}
	}
	if (Stabilizer_chain) {
		if (f_v) {
			cout << "action::report we have Stabilizer_chain" << endl;
		}
		if (f_strong_gens) {

			if (f_v) {
				cout << "action::report we have f_strong_gens" << endl;
			}

			action_global Global;

			if (f_v) {
				cout << "action::report before Global.report_strong_generators" << endl;
			}
			Global.report_strong_generators(
					ost,
					LG_Draw_options,
					SG,
					this,
					verbose_level);
			if (f_v) {
				cout << "action::report after Global.report_strong_generators" << endl;
			}


		}
	}
	if (f_v) {
		cout << "action::report done" << endl;
	}
}



void action::report_group_name_and_degree(
		std::ostream &ost,
		graphics::layered_graph_draw_options *LG_Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::report_group_name_and_degree" << endl;
	}

	if (label_tex.length() == 0) {
		cout << "action::report_group_name_and_degree the group has no tex-name" << endl;
		exit(1);
	}
	ost << "Group action $" << label_tex
			<< "$ of degree " << degree << "\\\\" << endl;


}

void action::report_type_of_action(
		std::ostream &ost,
		graphics::layered_graph_draw_options *O,
		int verbose_level)
{
	std::string txt;
	std::string tex;
	action_global AcGl;

	AcGl.get_symmetry_group_type_text(txt, tex, type_G);


	ost << "The action is of type " << tex << "\\\\" << endl;

	ost << "\\bigskip" << endl;

}

void action::report_what_we_act_on(
		std::ostream &ost,
		graphics::layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::report_what_we_act_on" << endl;
	}


	report_type_of_action(ost, O, verbose_level);

	if (is_matrix_group()) {

		if (f_v) {
			cout << "action::report_what_we_act_on is_matrix_group is true" << endl;
		}
		field_theory::finite_field *F;
		algebra::matrix_group *M;

		M = get_matrix_group();
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

		if (type_G == action_on_orthogonal_t) {

			if (G.AO->f_on_points) {
				ost << "acting on points only\\\\" << endl;
				ost << "Number of points = " << G.AO->O->Hyperbolic_pair->nb_points << "\\\\" << endl;
			}
			else if (G.AO->f_on_lines) {
				ost << "acting on lines only\\\\" << endl;
				ost << "Number of lines = " << G.AO->O->Hyperbolic_pair->nb_lines << "\\\\" << endl;
			}
			else if (G.AO->f_on_points_and_lines) {
				ost << "acting on points and lines\\\\" << endl;
				ost << "Number of points = " << G.AO->O->Hyperbolic_pair->nb_points << "\\\\" << endl;
				ost << "Number of lines = " << G.AO->O->Hyperbolic_pair->nb_lines << "\\\\" << endl;
			}

			G.AO->O->Quadratic_form->report_quadratic_form(
					ost, 0 /* verbose_level */);

			ost << "Tactical decomposition induced by a hyperbolic pair:\\\\" << endl;
			G.AO->O->report_schemes_easy(ost);

			G.AO->O->report_points(ost, 0 /* verbose_level */);

			G.AO->O->report_lines(ost, 0 /* verbose_level */);

		}

		if (M->f_projective) {

			ost << "Group Action $" << label_tex
					<< "$ on Projective Space ${\\rm PG}"
							"(" << M->n - 1 << ", " << F->q << ")$\\\\" << endl;

		}
		else if (M->f_affine) {

			ost << "Group Action $" << label_tex
					<< "$ on Affine Space ${\\rm AG}"
							"(" << M->n << ", " << F->q << ")$\\\\" << endl;

		}
		else if (M->f_general_linear) {

			ost << "Group Action $" << label_tex
					<< "$ on Affine Space ${\\rm AG}"
							"(" << M->n << ", " << F->q << ")$\\\\" << endl;

		}


		ost << "The finite field ${\\mathbb F}_{" << F->q << "}$:\\\\" << endl;

		F->Io->cheat_sheet(ost, verbose_level);

		ost << endl << "\\bigskip" << endl << endl;


	}

#if 0
	if (degree < 1000) {
		ost << "The group acts on the following set of size " << degree << ":\\\\" << endl;

		if (ptr->ptr_unrank_point) {
			if (f_v) {
				cout << "action::report_what_we_act_on before latex_all_points" << endl;
			}
			latex_all_points(ost);
			if (f_v) {
				cout << "action::report_what_we_act_on after latex_all_points" << endl;
			}
		}
		else {
			ost << "we don't have an unrank point function\\\\" << endl;
		}
	}
#endif



	if (f_v) {
		cout << "action::report_what_we_act_on done" << endl;
	}
}



void action::list_elements_as_permutations_vertically(
		data_structures_groups::vector_ge *gens,
		ostream &ost)
{
	int i, j, a, len;

	len = gens->len;
	for (j = 0; j < len; j++) {
		ost << " & \\alpha_{" << j << "}";
	}
	ost << "\\\\" << endl;
	for (i = 0; i < degree; i++) {
		ost << setw(3) << i;
		for (j = 0; j < len; j++) {
			a = Group_element->element_image_of(i,
					gens->ith(j), 0 /* verbose_level */);
			ost << " & " << setw(3) << a;
		}
		ost << "\\\\" << endl;
	}
}

void action::print_symmetry_group_type(
		std::ostream &ost)
{
	action_global AG;

	AG.action_print_symmetry_group_type(ost, type_G);
	if (f_has_subaction) {
		ost << "->";
		subaction->print_symmetry_group_type(ost);
	}
	//else {
		//ost << "no subaction";
	//}

}


void action::print_info()
{
	cout << "ACTION " << label << " : " << label_tex
			<< " degree=" << degree << " of type ";
	print_symmetry_group_type(cout);
	cout << endl;
	cout << "low_level_point_size=" << low_level_point_size;
	cout << ", f_has_sims=" << f_has_sims;
	cout << ", f_has_strong_generators=" << f_has_strong_generators;
	cout << endl;
	cout << "make_element_size=" << make_element_size << endl;
	cout << "elt_size_in_int=" << elt_size_in_int << endl;

	if (f_is_linear) {
		cout << "linear of dimension " << dimension << endl;
		}
	else {
		cout << "the action is not linear" << endl;
	}

	if (Stabilizer_chain) {
		if (base_len()) {
			cout << "base: ";
			Lint_vec_print(cout, get_base(), base_len());
			cout << endl;
		}
	}
	else {
		cout << "The action does not have a stabilizer chain" << endl;
	}
	if (f_has_sims) {
		cout << "has sims" << endl;
		ring_theory::longinteger_object go;

		Sims->group_order(go);
		cout << "Order " << go << " = ";
		//int_vec_print(cout, Sims->orbit_len, base_len());
		for (int t = 0; t < base_len(); t++) {
			cout << Sims->get_orbit_length(t);
			if (t < base_len() - 1) {
				cout << " * ";
			}
		}
		//cout << endl;
	}
	cout << endl;

}

void action::report_basic_orbits(
		std::ostream &ost)
{
	int i;

	if (Stabilizer_chain) {
		ost << "The base has length " << base_len() << "\\\\" << endl;
		ost << "The basic orbits are: \\\\" << endl;
		for (i = 0; i < base_len(); i++) {
			ost << "Basic orbit " << i << " is orbit of " << base_i(i)
				<< " of length " << transversal_length_i(i) << "\\\\" << endl;
		}
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
	}
}

void action::print_base()
{
	if (Stabilizer_chain) {
		cout << "action " << label << " has base ";
		Lint_vec_print(cout, get_base(), base_len());
		cout << endl;
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
	}
}

void action::print_bare_base(
		std::ofstream &ost)
{
	if (Stabilizer_chain) {
		orbiter_kernel_system::Orbiter->Lint_vec->print_bare_fully(ost, get_base(), base_len());
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
		exit(1);
	}
}

void action::latex_all_points(
		std::ostream &ost)
{
	int i;
	int *v;


	if (ptr->ptr_unrank_point == NULL) {
		cout << "action::latex_all_points ptr->ptr_unrank_point == NULL" << endl;
		return;
	}
	v = NEW_int(low_level_point_size);
#if 0
	cout << "action::latex_all_points "
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
	cout << "action::latex_all_points done" << endl;
#else
	if (low_level_point_size < 10) {
		ost << "\\begin{multicols}{2}" << endl;
	}
	ost << "\\noindent" << endl;
	for (i = 0; i < degree; i++) {
		Group_element->unrank_point(i, v);
		ost << i << " = ";
		Int_vec_print(ost, v, low_level_point_size);
		ost << "\\\\" << endl;
	}
	if (low_level_point_size < 10) {
		ost << "\\end{multicols}" << endl;
	}

#endif

	FREE_int(v);
}

void action::latex_point_set(
		std::ostream &ost,
		long int *set, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *v;

	if (f_v) {
		cout << "action::print_points "
				"low_level_point_size=" << low_level_point_size <<  endl;
	}
	v = NEW_int(low_level_point_size);
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


	if (ptr->ptr_unrank_point) {
		if (low_level_point_size < 10) {
			ost << "\\begin{multicols}{2}" << endl;
		}
		ost << "\\noindent" << endl;
		for (i = 0; i < sz; i++) {
			Group_element->unrank_point(set[i], v);
			ost << i << " : ";
			ost << set[i] << " = ";
			Int_vec_print(ost, v, low_level_point_size);
			ost << "\\\\" << endl;
		}
		if (low_level_point_size < 10) {
			ost << "\\end{multicols}" << endl;
		}
	}
#endif

	FREE_int(v);
	if (f_v) {
		cout << "action::print_points done" << endl;
	}
}


void action::print_group_order(
		std::ostream &ost)
{
	ring_theory::longinteger_object go;
	group_order(go);
	cout << go;
}

void action::print_group_order_long(
		std::ostream &ost)
{
	int i;

	ring_theory::longinteger_object go;
	group_order(go);
	cout << go << " =";
	if (Stabilizer_chain) {
		for (i = 0; i < base_len(); i++) {
			cout << " " << transversal_length_i(i);
		}
	}
	else {
		cout << "action " << label << " does not have a base" << endl;
	}

}

void action::print_vector(
		data_structures_groups::vector_ge &v)
{
	int i, l;

	l = v.len;
	cout << "vector of " << l << " group elements:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " : " << endl;
		Group_element->element_print_quick(v.ith(i), cout);
		cout << endl;
	}
}

void action::print_vector_as_permutation(
		data_structures_groups::vector_ge &v)
{
	int i, l;

	l = v.len;
	cout << "vector of " << l << " group elements:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " : ";
		Group_element->element_print_as_permutation(v.ith(i), cout);
		cout << endl;
	}
}


void action::write_set_of_elements_latex_file(
		std::string &fname,
		std::string &title, int *Elt, int nb_elts)
{
	{
		ofstream ost(fname);
		number_theory::number_theory_domain NT;

		string author, extra_praeamble;

		l1_interfaces::latex_interface L;

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
			Group_element->element_print_latex(Elt + i * elt_size_in_int, ost);
			ost << "$$" << endl;
		}

		L.foot(ost);


	}

	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

}

void action::export_to_orbiter(
		std::string &fname, std::string &label,
		groups::strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	long int a;
	orbiter_kernel_system::file_io Fio;
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "action::export_to_orbiter" << endl;
	}

	SG->group_order(go);
	if (f_v) {
		cout << "action::export_to_orbiter go = " << go << endl;
		cout << "action::export_to_orbiter number of generators = " << SG->gens->len << endl;
		cout << "action::export_to_orbiter degree = " << degree << endl;
	}
	{
		ofstream fp(fname);

		fp << "GENERATORS_" << label << " = \\" << endl;
		for (i = 0; i < SG->gens->len; i++) {
			fp << "\t\"";
			for (j = 0; j < degree; j++) {
				if (false) {
					cout << "action::export_to_orbiter computing image of " << j << " under generator " << i << endl;
				}
				a = Group_element->element_image_of(j, SG->gens->ith(i), 0 /* verbose_level*/);
				fp << a;
				if (j < degree - 1 || i < SG->gens->len - 1) {
					fp << ",";
				}
			}
			fp << "\"";
			if (i < SG->gens->len - 1) {
				fp << "\\" << endl;
			}
			else {
				fp << endl;
			}
		}

		fp << endl;
		fp << label << ":" << endl;
		fp << "\t$(ORBITER_PATH)orbiter.out -v 2 \\" << endl;
		fp << "\t-define G -permutation_group -symmetric_group " << degree << " \\" << endl;
		fp << "\t-subgroup_by_generators " << label << " " << go << " " << SG->gens->len << " $(GENERATORS_" << label << ") -end \\" << endl;

		//		$(ORBITER_PATH)orbiter.out -v 10
		//			-define G -permutation_group -symmetric_group 13
		//				-subgroup_by_generators H5 5 1 $(GENERATORS_H5) -end
		// with backslashes at the end of the line

	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "action::export_to_orbiter" << endl;
	}
}


void action::export_to_orbiter_as_bsgs(
		std::string &fname,
		std::string &label,
		std::string &label_tex,
		groups::strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs" << endl;
	}


	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs "
				"before SG->export_to_orbiter_as_bsgs" << endl;
	}
	SG->export_to_orbiter_as_bsgs(
			this,
			fname, label, label_tex,
			verbose_level);
	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs "
				"after SG->export_to_orbiter_as_bsgs" << endl;
	}


	if (f_v) {
		cout << "action::export_to_orbiter_as_bsgs" << endl;
	}
}


void action::print_one_element_tex(
		std::ostream &ost,
		int *Elt, int f_with_permutation)
{
	ost << "$$" << endl;
	Group_element->element_print_latex(Elt, ost);
	ost << "$$" << endl;

	if (f_with_permutation) {
		Group_element->element_print_as_permutation(Elt, ost);
		ost << "\\\\" << endl;

		int *perm;
		int h, j;

		perm = NEW_int(degree);

		Group_element->compute_permutation(
				Elt,
				perm, 0 /* verbose_level */);

		for (h = 0; h < degree; h++) {
			j = perm[h];

			ost << j;
			if (j < degree) {
				ost << ", ";
			}
		}
		ost << "\\\\" << endl;

		FREE_int(perm);

	}

}


}}}

