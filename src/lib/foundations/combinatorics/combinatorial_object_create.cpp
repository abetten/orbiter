/*
 * combinatorial_object_create.cpp
 *
 *  Created on: Nov 9, 2019
 *      Author: anton
 */





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


combinatorial_object_create::combinatorial_object_create()
{
	F = NULL;
	//A = NULL;
	//f_has_group = FALSE;
	//Sg = NULL;

	//char fname[1000];
	nb_pts = 0;
	Pts = NULL;

	null();
}

combinatorial_object_create::~combinatorial_object_create()
{
	freeself();
}

void combinatorial_object_create::null()
{
}

void combinatorial_object_create::freeself()
{
	if (F) {
		delete F;
		}
	if (Pts) {
		FREE_lint(Pts);
		}
#if 0
	if (Sg) {
		delete Sg;
		}
#endif
	null();
}

void combinatorial_object_create::init(combinatorial_object_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;


	if (f_v) {
		cout << "combinatorial_object_create::init" << endl;
		}
	combinatorial_object_create::Descr = Descr;
	if (!Descr->f_q) {
		cout << "combinatorial_object_create::init !Descr->f_q" << endl;
		exit(1);
		}
	q = Descr->q;
	if (f_v) {
		cout << "combinatorial_object_create::init q = " << q << endl;
		}
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);




	if (Descr->f_hyperoval) {
		F->create_hyperoval(
				Descr->f_translation, Descr->translation_exponent,
				Descr->f_Segre, Descr->f_Payne, Descr->f_Cherowitzo, Descr->f_OKeefe_Penttila,
				fname, nb_pts, Pts,
			verbose_level);

		F->export_magma(3, Pts, nb_pts, fname);
		F->export_gap(3, Pts, nb_pts, fname);

	}
	else if (Descr->f_subiaco_oval) {
		F->create_subiaco_oval(
				Descr->f_short,
			fname, nb_pts, Pts,
			verbose_level);


		F->export_magma(3, Pts, nb_pts, fname);
		F->export_gap(3, Pts, nb_pts, fname);


	}
	else if (Descr->f_subiaco_hyperoval) {
		F->create_subiaco_hyperoval(
			fname, nb_pts, Pts,
			verbose_level);


		F->export_magma(3, Pts, nb_pts, fname);
		F->export_gap(3, Pts, nb_pts, fname);

	}
	else if (Descr->f_adelaide_hyperoval) {

		finite_field *FQ;
		subfield_structure *S;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Descr->Q, Descr->poly_Q, 0);

		S = NEW_OBJECT(subfield_structure);
		S->init(FQ, F, verbose_level);

		S->create_adelaide_hyperoval(
			fname, nb_pts, Pts,
			verbose_level);


		F->export_magma(3, Pts, nb_pts, fname);
		F->export_gap(3, Pts, nb_pts, fname);


		FREE_OBJECT(S);
		FREE_OBJECT(FQ);
	}
	else if (Descr->f_BLT_database) {
		F->create_BLT_from_database(Descr->f_BLT_in_PG /* f_embedded */, Descr->BLT_k,
			fname, nb_pts, Pts,
			verbose_level);
	}
#if 0
	else if (f_BLT_Linear) {
		create_BLT(f_BLT_in_PG /* f_embedded */, F,
			TRUE /* f_Linear */,
			FALSE /* f_Fisher */,
			FALSE /* f_Mondello */,
			FALSE /* f_FTWKB */,
			f_poly, poly,
			f_poly_Q, poly_Q,
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (f_BLT_Fisher) {
		create_BLT(f_BLT_in_PG /* f_embedded */, q,
			FALSE /* f_Linear */,
			TRUE /* f_Fisher */,
			FALSE /* f_Mondello */,
			FALSE /* f_FTWKB */,
			f_poly, poly,
			f_poly_Q, poly_Q,
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (f_BLT_Mondello) {
		create_BLT(f_BLT_in_PG /* f_embedded */, q,
			FALSE /* f_Linear */,
			FALSE /* f_Fisher */,
			TRUE /* f_Mondello */,
			FALSE /* f_FTWKB */,
			f_poly, poly,
			f_poly_Q, poly_Q,
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (f_BLT_FTWKB) {
		create_BLT(f_BLT_in_PG /* f_embedded */, q,
			FALSE /* f_Linear */,
			FALSE /* f_Fisher */,
			FALSE /* f_Mondello */,
			TRUE /* f_FTWKB */,
			f_poly, poly,
			f_poly_Q, poly_Q,
			fname, nb_pts, Pts,
			verbose_level);
	}
#endif
	else if (Descr->f_ovoid) {
		F->create_ovoid(
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (Descr->f_Baer) {
		if (!Descr->f_n) {
			cout << "please specify the projective dimension "
					"using the option -n <n>" << endl;
			exit(1);
			}

		if (!Descr->f_Q) {
			cout << "please specify the field order "
					"using the option -Q <Q>" << endl;
			exit(1);
			}

		finite_field *FQ;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Descr->Q, Descr->poly_Q, 0);

		FQ->create_Baer_substructure(Descr->n, F,
			fname, nb_pts, Pts,
			verbose_level);
		FREE_OBJECT(FQ);
	}
	else if (Descr->f_orthogonal) {
		if (!Descr->f_n) {
			cout << "please specify the projective dimension "
					"using the option -n <n>" << endl;
			exit(1);
			}
		F->create_orthogonal(Descr->orthogonal_epsilon, Descr->n,
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (Descr->f_hermitian) {
		if (!Descr->f_n) {
			cout << "please specify the projective dimension "
					"using the option -n <n>" << endl;
			exit(1);
			}
		F->create_hermitian(Descr->n,
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (Descr->f_cubic) {
		F->create_cubic(
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (Descr->f_twisted_cubic) {
		F->create_twisted_cubic(
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (Descr->f_elliptic_curve) {
		F->create_elliptic_curve(
				Descr->elliptic_curve_b, Descr->elliptic_curve_c,
			fname, nb_pts, Pts,
			verbose_level);
	}
#if 0
	else if (Descr->f_Hill_cap_56) {
		Hill_cap56(
			fname, nb_pts, Pts,
			verbose_level);
	}
#endif
	else if (Descr->f_ttp_code) {

		if (!Descr->f_Q) {
			cout << "please specify the field order "
					"using the option -Q <Q>" << endl;
			exit(1);
			}

		finite_field *FQ;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Descr->Q, Descr->poly_Q, 0);
		FQ->create_ttp_code(F,
				Descr->f_ttp_construction_A, Descr->f_ttp_hyperoval, Descr->f_ttp_construction_B,
			fname, nb_pts, Pts,
			verbose_level);
		FREE_OBJECT(FQ);
	}
	else if (Descr->f_unital_XXq_YZq_ZYq) {
		F->create_unital_XXq_YZq_ZYq(
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (Descr->f_desarguesian_line_spread_in_PG_3_q) {

		if (!Descr->f_Q) {
			cout << "please specify the field order "
					"using the option -Q <Q>" << endl;
			exit(1);
			}

		finite_field *FQ;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Descr->Q, Descr->poly_Q, 0);

		FQ->create_desarguesian_line_spread_in_PG_3_q(F,
				Descr->f_embedded_in_PG_4_q,
			fname, nb_pts, Pts,
			verbose_level);
		FREE_OBJECT(FQ);

	}
	else if (Descr->f_Buekenhout_Metz) {

		finite_field *FQ;
		geometry_global Gg;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Descr->Q, Descr->poly_Q, 0);

		Gg.create_Buekenhout_Metz(F, FQ,
				Descr->f_classical, Descr->f_Uab, Descr->parameter_a, Descr->parameter_b,
			fname, nb_pts, Pts,
			verbose_level);

	}
	else if (Descr->f_whole_space) {
		if (!Descr->f_n) {
			cout << "please specify the projective dimension "
					"using the option -n <n>" << endl;
			exit(1);
			}
		F->create_whole_space(Descr->n,
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (Descr->f_hyperplane) {
		if (!Descr->f_n) {
			cout << "please specify the projective dimension "
					"using the option -n <n>" << endl;
			exit(1);
			}
		F->create_hyperplane(Descr->n,
				Descr->pt,
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (Descr->f_segre_variety) {
		F->create_segre_variety(Descr->segre_variety_a, Descr->segre_variety_b,
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (Descr->f_Maruta_Hamada_arc) {
		F->create_Maruta_Hamada_arc(
			fname, nb_pts, Pts,
			verbose_level);
	}
	else if (Descr->f_projective_variety) {
		F->create_projective_variety(
				Descr->variety_label,
				Descr->n + 1, Descr->variety_degree,
				Descr->variety_coeffs,
				Descr->Monomial_ordering_type,
				fname, nb_pts, Pts,
				verbose_level);
	}
	else if (Descr->f_projective_curve) {
		F->create_projective_curve(
				Descr->curve_label,
				Descr->curve_nb_vars, Descr->curve_degree,
				Descr->curve_coeffs,
				Descr->Monomial_ordering_type,
				fname, nb_pts, Pts,
				verbose_level);
	}
	else {
		cout << "combinatorial_object_create::init nothing to create" << endl;
		exit(1);
		}


	if (f_v) {
		cout << "combinatorial_object_create::init set = ";
		lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
		}

#if 0
	set = NEW_lint(nb_pts);
	lint_vec_copy(Pts, set, nb_pts);
	set_size = nb_pts;

	FREE_lint(Pts);

	if (f_has_group) {
		cout << "combinatorial_object_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
		}
#endif



	if (f_v) {
		cout << "combinatorial_object_create::init done" << endl;
	}
}

#if 0
void combinatorial_object_create::apply_transformations(const char **transform_coeffs,
	int *f_inverse_transform, int nb_transform, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_create::apply_transformations done" << endl;
	}
}
#endif


}}


