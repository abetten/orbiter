/*
 * orthogonal_space_with_action.cpp
 *
 *  Created on: Jan 12, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


orthogonal_space_with_action::orthogonal_space_with_action()
{
	Record_birth();
	Descr = NULL;
	P = NULL;
	O = NULL;
	f_semilinear = false;
	A = NULL;
	AO = NULL;
	Blt_set_domain_with_action = NULL;
}

orthogonal_space_with_action::~orthogonal_space_with_action()
{
	Record_death();
	if (O) {
		FREE_OBJECT(O);
	}
	if (A) {
		FREE_OBJECT(A);
	}
	if (Blt_set_domain_with_action) {
		FREE_OBJECT(Blt_set_domain_with_action);
	}
}

void orthogonal_space_with_action::init(
		orthogonal_space_with_action_description *Descr,
		int verbose_level)
// creates a projective space and an orthogonal space.
// For n == 5, it also creates a blt_set_domain
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::init" << endl;
	}
	orthogonal_space_with_action::Descr = Descr;

	P = NEW_OBJECT(geometry::projective_geometry::projective_space);

	if (f_v) {
		cout << "orthogonal_space_with_action::init "
				"before P->projective_space_init" << endl;
	}

	P->projective_space_init(
			Descr->n - 1, Descr->F,
		false /* f_init_incidence_structure */,
		verbose_level);

	if (f_v) {
		cout << "orthogonal_space_with_action::init "
				"after P->projective_space_init" << endl;
	}

	O = NEW_OBJECT(geometry::orthogonal_geometry::orthogonal);


	if (f_v) {
		cout << "orthogonal_space_with_action::init "
				"before O->init" << endl;
	}
	O->init(Descr->epsilon, Descr->n, Descr->F, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal_space_with_action::init "
				"after O->init" << endl;
	}


	if (Descr->f_label_txt) {
		O->label_txt.assign(Descr->label_txt);
	}
	if (Descr->f_label_tex) {
		O->label_tex.assign(Descr->label_tex);
	}




	if (!Descr->f_without_group) {

		if (f_v) {
			cout << "orthogonal_space_with_action::init "
					"before init_group" << endl;
		}
		init_group(verbose_level - 2);
		if (f_v) {
			cout << "orthogonal_space_with_action::init "
					"after init_group" << endl;
		}
	}
	else {
		if (f_v) {
			cout << "orthogonal_space_with_action::init without group" << endl;
		}

	}

	if (Descr->n == 5) {

		if (f_v) {
			cout << "orthogonal_space_with_action::init "
					"allocating Blt_Set_domain" << endl;
		}
		Blt_set_domain_with_action = NEW_OBJECT(
				orthogonal_geometry_applications::blt_set_domain_with_action);


		if (f_v) {
			cout << "orthogonal_space_with_action::init "
					"before Blt_set_domain_with_action->init" << endl;
		}
		Blt_set_domain_with_action->init(
				A, P, O, Descr->f_create_extension_fields,
				verbose_level);
		if (f_v) {
			cout << "orthogonal_space_with_action::init_blt_set_domain "
					"after Blt_set_domain_with_action->init" << endl;
		}
	}


	if (f_v) {
		cout << "orthogonal_space_with_action::init done" << endl;
	}
}

void orthogonal_space_with_action::init_group(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::init_group" << endl;
	}

	algebra::number_theory::number_theory_domain NT;

	f_semilinear = true;
	if (NT.is_prime(Descr->F->q)) {
		f_semilinear = false;
	}


	A = NEW_OBJECT(actions::action);

	if (f_v) {
		cout << "orthogonal_space_with_action::init_group before "
				"A->Known_groups->init_orthogonal_group_with_O" << endl;
	}

	A->Known_groups->init_orthogonal_group_with_O(
			O,
			true /* f_on_points */,
			false /* f_on_lines */,
			false /* f_on_points_and_lines */,
			f_semilinear,
			true /* f_basis */,
			verbose_level - 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"after A->Known_groups->init_orthogonal_group_with_O" << endl;
	}

	if (f_v) {
		cout << "A->make_element_size = "
			<< A->make_element_size << endl;
		cout << "orthogonal_space_with_action::init_group "
				"degree = " << A->degree << endl;
	}

	if (!A->f_has_sims) {
		cout << "orthogonal_space_with_action::init_group "
				"!A->f_has_sims" << endl;
		exit(1);
	}

#if 0
	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"before A->lex_least_base_in_place" << endl;
	}
	A->lex_least_base_in_place(A->Sims, verbose_level - 2);
	if (f_v) {
		cout << "orthogonal_space_with_action::init_group "
				"after A->lex_least_base_in_place" << endl;
	}
	if (f_v) {
		cout << "orthogonal_space_with_action::init_group base: ";
		Lint_vec_print(cout, A->get_base(), A->base_len());
		cout << endl;
	}
#endif


	AO = A->G.AO;
	O = AO->O;


	if (f_v) {
		cout << "orthogonal_space_with_action::init_group done" << endl;
	}
}

void orthogonal_space_with_action::report(
		other::graphics::layered_graph_draw_options *Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::report" << endl;
	}

	{
		string fname_report;
		fname_report = O->label_txt + "_report.tex";
		other::l1_interfaces::latex_interface L;
		other::orbiter_kernel_system::file_io Fio;

		{
			ofstream ost(fname_report);
			L.head_easy(ost);

			if (f_v) {
				cout << "orthogonal_space_with_action::report "
						"before report2" << endl;
			}
			report2(ost, Draw_options, verbose_level);
			if (f_v) {
				cout << "orthogonal_space_with_action::report "
						"after report2" << endl;
			}

			L.foot(ost);
		}

		cout << "Written file " << fname_report << " of size "
				<< Fio.file_size(fname_report) << endl;
	}

}

void orthogonal_space_with_action::report2(
		std::ostream &ost,
		other::graphics::layered_graph_draw_options *Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::report2" << endl;
	}


	if (!Descr->f_without_group) {
		if (f_v) {
			cout << "orthogonal_space_with_action::report2 "
					"before A>report" << endl;
		}

		A->report(ost,
				false /* f_sims */, NULL,
				false /* f_strong_gens */, NULL,
				Draw_options,
				verbose_level - 1);

		if (f_v) {
			cout << "orthogonal_space_with_action::report2 "
					"after A->report" << endl;
		}
	}
	else {
		ost << "The group is not available.\\\\" << endl;
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::report2 before O->report" << endl;
	}
	O->report(ost, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::report2 after O->report" << endl;
	}


	if (f_v) {
		cout << "orthogonal_space_with_action::report2 done" << endl;
	}
}


void orthogonal_space_with_action::make_table_of_blt_sets(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets" << endl;
	}

	if (O->Quadratic_form->n != 5) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets "
				"we need a five-dimensional orthogonal space" << endl;
		exit(1);
	}

	table_of_blt_sets *T;

	T = NEW_OBJECT(table_of_blt_sets);

	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets "
				"before T->init" << endl;
	}
	T->init(this, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets "
				"after T->init" << endl;
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets "
				"before T->do_export" << endl;
	}
	T->do_export(verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets "
				"after T->do_export" << endl;
	}

	FREE_OBJECT(T);


	if (f_v) {
		cout << "orthogonal_space_with_action::make_table_of_blt_sets done" << endl;
	}

}

void orthogonal_space_with_action::create_perp_of_point(
		long int pt_rank_pg,
		std::vector<long int> &Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_perp_of_point" << endl;
	}


	geometry::projective_geometry::projective_space_basic Projective_space_basic;

	if (f_v) {
		cout << "orthogonal_space_with_action::create_perp_of_point "
				"before Projective_space_basic.init" << endl;
	}
	Projective_space_basic.init(P->Subspaces->F, verbose_level - 1);
	if (f_v) {
		cout << "orthogonal_space_with_action::create_perp_of_point "
				"after Projective_space_basic.init" << endl;
	}


	int d, k;
	int *v;
	int *Perp;
	int *message;
	int *word;

	d = P->Subspaces->n + 1;
	k = d - 1;

	v = NEW_int(d);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_perp_of_point "
				"v = ";
		Int_vec_print(cout, v, d);
		cout << endl;
	}

	Perp = NEW_int(d * d);

	combinatorics::other_combinatorics::combinatorics_domain Combi;

	int nb_messages;
	int h;

	nb_messages = Combi.generalized_binomial(k, 1, P->Subspaces->F->q);
	if (f_v) {
		cout << "orthogonal_space_with_action::create_perp_of_point "
				"nb_messages = " << nb_messages << endl;
	}

	message = NEW_int(k);
	word = NEW_int(d);


	P->Subspaces->unrank_point(v, pt_rank_pg);

	O->Quadratic_form->perp_of_point(v, Perp, verbose_level - 1);
	if (f_v) {
		cout << "orthogonal_space_with_action::create_perp_of_point "
				"Perp = " << endl;
		Int_matrix_print(Perp, k, d);
	}



	long int rk;
	int form_value;

	if (f_v) {
		cout << "projective_space_basic::all_PG_elements_in_subspace_array_is_given" << endl;
	}

	for (h = 0; h < nb_messages; h++) {

		if (f_v) {
			cout << "orthogonal_space_with_action::create_perp_of_point "
					"message " << h << " / " << nb_messages << endl;
		}

		Projective_space_basic.PG_element_unrank_modified(message, 1, k, h);

		if (f_v) {
			cout << "orthogonal_space_with_action::create_perp_of_point "
					"message " << h << " / " << nb_messages << " is ";
			Int_vec_print(cout, message, k);
			cout << endl;
		}

		P->Subspaces->F->Linear_algebra->mult_vector_from_the_left(
				message, Perp, word, k, d);

		if (f_v) {
			cout << "orthogonal_space_with_action::create_perp_of_point yields word ";
			Int_vec_print(cout, word, d);
			cout << endl;
		}

		if (f_v) {
			cout << "orthogonal_space_with_action::create_perp_of_point "
					"before O->Quadratic_form->evaluate_quadratic_form" << endl;
		}

		form_value = O->Quadratic_form->evaluate_quadratic_form(
						word, 1 /* stride */);

		if (f_v) {
			cout << "orthogonal_space_with_action::create_perp_of_point "
					"after O->Quadratic_form->evaluate_quadratic_form" << endl;
		}
		if (f_v) {
			cout << "orthogonal_space_with_action::create_perp_of_point "
					"form_value = " << form_value << " not on the quadric" << endl;
		}

		if (form_value == 0) {

			if (f_v) {
				cout << "orthogonal_space_with_action::create_perp_of_point "
						"the point lies on the quadric" << endl;
			}

			O->Orthogonal_indexing->Sbar_rank(word, 1 /* stride */, d / 2, rk, 0 /*verbose_level - 1*/);

#if 0
			rk = O->Hyperbolic_pair->rank_point(
					word, 1 /* stride */, verbose_level - 1);
#endif

			if (f_v) {
				cout << "orthogonal_space_with_action::create_perp_of_point "
						"rk = " << rk << endl;
			}

			Pts.push_back(rk);

		}
		else {
			if (f_v) {
				cout << "orthogonal_space_with_action::create_perp_of_point "
						"form_value = " << form_value << " not on the quadric" << endl;
			}
		}

	}

	if (f_v) {
		cout << "orthogonal_space_with_action::create_perp_of_point "
				"perp of point " << pt_rank_pg << " is the following set of size " << Pts.size() << ":" << endl;
		Lint_vec_stl_print(cout, Pts);
		cout << endl;
	}



	FREE_int(v);
	FREE_int(Perp);
	FREE_int(message);
	FREE_int(word);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_perp_of_point" << endl;
	}
}

void orthogonal_space_with_action::create_orthogonal_reflections(
		long int *pts, int nb_pts,
		data_structures_groups::vector_ge *&vec,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections" << endl;
	}
	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections "
				"nb_pts = " << nb_pts << endl;
	}


	int *z;
	int *Data;
	int i, d, sz;

	d = P->Subspaces->n + 1;

	sz = d * d;
	if (f_semilinear) {
		sz++;
	}

	z = NEW_int(d);
	Data = NEW_int(nb_pts * sz);
	for (i = 0; i < nb_pts; i++) {

		if (f_v) {
			cout << "orthogonal_space_with_action::create_orthogonal_reflections "
					"i = " << i << " / " << nb_pts << endl;
		}
		P->Subspaces->unrank_point(z, pts[i]);

		O->Orthogonal_group->make_orthogonal_reflection(
				Data + i * sz, z, verbose_level);

		if (f_semilinear) {
			Data[i * sz + d * d] = 0;
		}

	}


	vec = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections "
				"A = " << endl;
		A->print_info();
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections "
				"before vec->init_from_data" << endl;
	}
	vec->init_from_data(
			A, Data,
			nb_pts, sz, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections "
				"after vec->init_from_data" << endl;
	}

	FREE_int(Data);
	FREE_int(z);



	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections done" << endl;
	}
}



void orthogonal_space_with_action::create_Siegel_transformation(
		int *u, int *v, int len,
		data_structures_groups::vector_ge *&vec,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_Siegel_transformation" << endl;
	}
	if (f_v) {
		cout << "orthogonal_space_with_action::create_Siegel_transformation "
				"u = ";
		Int_vec_print(cout, u, len);
		cout << endl;
		cout << "orthogonal_space_with_action::create_Siegel_transformation "
				"v = ";
		Int_vec_print(cout, v, len);
		cout << endl;
	}

	if (len != O->Quadratic_form->n) {
		cout << "orthogonal_space_with_action::create_Siegel_transformation "
				"len != O->Quadratic_form->n" << endl;
		exit(1);
	}

	int nb_pts = 1;

	int *z;
	int *Data;
	int i, d, sz;

	d = P->Subspaces->n + 1;

	sz = d * d;
	if (f_semilinear) {
		sz++;
	}

	z = NEW_int(d);
	Data = NEW_int(nb_pts * sz);
	for (i = 0; i < nb_pts; i++) {

		if (f_v) {
			cout << "orthogonal_space_with_action::create_Siegel_transformation "
					"i = " << i << " / " << nb_pts << endl;
		}
		//P->Subspaces->unrank_point(z, pts[i]);

		//O->Orthogonal_group->make_orthogonal_reflection(
		//		Data + i * sz, z, verbose_level);

		O->Orthogonal_group->make_Siegel_Transformation(
				Data + i * sz, v, u,
			len /* n */, O->Quadratic_form->Gram_matrix, verbose_level - 2);
		// if u is singular and v \in \la u \ra^\perp, then
		// \rho_{u,v}(x) := x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
		// is called Siegel transform (see Taylor p. 148)
		// Here Q is the quadratic form and \beta is
		// the corresponding bilinear form


		if (f_semilinear) {
			Data[i * sz + d * d] = 0;
		}

	}


	vec = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_Siegel_transformation "
				"A = " << endl;
		A->print_info();
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::create_Siegel_transformation "
				"before vec->init_from_data" << endl;
	}
	vec->init_from_data(
			A, Data,
			nb_pts, sz, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::create_Siegel_transformation "
				"after vec->init_from_data" << endl;
	}

	FREE_int(Data);
	FREE_int(z);



	if (f_v) {
		cout << "orthogonal_space_with_action::create_Siegel_transformation done" << endl;
	}
}




void orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4(
		long int *pts, int nb_pts,
		actions::action *A4,
		data_structures_groups::vector_ge *&vec6,
		data_structures_groups::vector_ge *&vec4,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4" << endl;
	}
	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
				"nb_pts = " << nb_pts << endl;
	}


	int *z;
	int *Data6;
	int i, d, sz;

	d = P->Subspaces->n + 1;
	if (d != 6) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 d != 6" << endl;
		exit(1);
	}

	sz = d * d;
	if (f_semilinear) {
		sz++;
	}

	z = NEW_int(d);
	Data6 = NEW_int(nb_pts * sz);
	for (i = 0; i < nb_pts; i++) {

		if (f_v) {
			cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
					"i = " << i << " / " << nb_pts << endl;
		}
		P->Subspaces->unrank_point(z, pts[i]);

		O->Orthogonal_group->make_orthogonal_reflection(
				Data6 + i * sz, z, verbose_level);

		if (f_semilinear) {
			Data6[i * sz + d * d] = 0;
		}

	}

	vec6 = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
				"A = " << endl;
		A->print_info();
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
				"before vec->init_from_data" << endl;
	}
	vec6->init_from_data(
			A, Data6,
			nb_pts, sz, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
				"after vec->init_from_data" << endl;
	}



	geometry::projective_geometry::klein_correspondence *K;
	//geometry::orthogonal_geometry::orthogonal *O;
	int sz4;

	sz4 = 4 * 4;

	if (f_semilinear) {
		sz4++;
	}

	int *Data4;

	Data4 = NEW_int(nb_pts * sz4);


	//F = A->matrix_group_finite_field();

	//O = NEW_OBJECT(geometry::orthogonal_geometry::orthogonal);
	//O->init(1 /* epsilon */, 6 /* n */, F, verbose_level);

	K = NEW_OBJECT(geometry::projective_geometry::klein_correspondence);
	K->init(O->F, O, verbose_level);


	for (i = 0; i < vec6->len; i++) {

		if (f_v) {
			cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
					"generator " << i << " / " << vec6->len << ":" << endl;
		}


		int f_has_polarity;

		//K->reverse_isomorphism(vec6->ith(i), Data4 + i * sz4, verbose_level);
		K->reverse_isomorphism_with_polarity(vec6->ith(i), Data4 + i * sz4, f_has_polarity, verbose_level);

		if (f_semilinear) {
			Data4[i * sz4 + 16] = 0;
		}

		if (f_v) {
			cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
					"before:" << endl;
			Int_matrix_print(vec6->ith(i), 6, 6);

			cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
					"after:" << endl;
			Int_matrix_print(Data4 + i * sz4, 4, 4);
			cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
					"f_has_polarity = " << f_has_polarity << endl;
		}

	}

	vec4 = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
				"A4 = " << endl;
		A4->print_info();
	}

	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
				"before vec4->init_from_data" << endl;
	}
	vec4->init_from_data(
			A4, Data4,
			nb_pts, sz4, verbose_level);
	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 "
				"after vec4->init_from_data" << endl;
	}



	FREE_OBJECT(K);
	//FREE_OBJECT(O);

	if (f_v) {
		cout << "orthogonal_space_with_action::create_orthogonal_reflections_6x6_and_4x4 done" << endl;
	}
}



}}}

