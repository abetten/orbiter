/*
 * geometric_object_create.cpp
 *
 *  Created on: Nov 9, 2019
 *      Author: anton
 */





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {


geometric_object_create::geometric_object_create()
{
	Descr = NULL;

	//std::string fname;
	nb_pts = 0;
	Pts = NULL;

	//std::string label_txt;
	//std::string label_tex;

}

geometric_object_create::~geometric_object_create()
{
	if (Pts) {
		FREE_lint(Pts);
	}
}

void geometric_object_create::init(geometric_object_description *Descr,
		projective_space *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;


	if (f_v) {
		cout << "geometric_object_create::init" << endl;
	}
	Descr->P = P;
	geometric_object_create::Descr = Descr;

	field_theory::finite_field *F;

	F = P->F;


	if (Descr->f_hyperoval) {
		if (f_v) {
			cout << "geometric_object_create::init before P->Arc_in_projective_space->create_hyperoval" << endl;
		}
		P->Arc_in_projective_space->create_hyperoval(
				Descr->f_translation,
				Descr->translation_exponent,
				Descr->f_Segre,
				Descr->f_Payne,
				Descr->f_Cherowitzo,
				Descr->f_OKeefe_Penttila,
				label_txt,
				label_tex,
				nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after P->Arc_in_projective_space->create_hyperoval" << endl;
		}

		//F->export_magma(3, Pts, nb_pts, fname);
		//F->export_gap(3, Pts, nb_pts, fname);

	}
	else if (Descr->f_subiaco_oval) {
		if (f_v) {
			cout << "geometric_object_create::init before P->Arc_in_projective_space->create_subiaco_oval" << endl;
		}
		P->Arc_in_projective_space->create_subiaco_oval(
				Descr->f_short,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after P->Arc_in_projective_space->create_subiaco_oval" << endl;
		}

		//F->export_magma(3, Pts, nb_pts, fname);
		//F->export_gap(3, Pts, nb_pts, fname);

	}
	else if (Descr->f_subiaco_hyperoval) {
		if (f_v) {
			cout << "geometric_object_create::init before P->Arc_in_projective_space->create_subiaco_hyperoval" << endl;
		}
		P->Arc_in_projective_space->create_subiaco_hyperoval(
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after P->Arc_in_projective_space->create_subiaco_hyperoval" << endl;
		}


		//F->export_magma(3, Pts, nb_pts, fname);
		//F->export_gap(3, Pts, nb_pts, fname);

	}
#if 0
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
#endif
	else if (Descr->f_BLT_database) {

		knowledge_base K;

		if (f_v) {
			cout << "geometric_object_create::init before K.retrieve_BLT_set_from_database" << endl;
		}
		K.retrieve_BLT_set_from_database(F,
				FALSE /* f_embedded */,
				Descr->BLT_database_k,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after K.retrieve_BLT_set_from_database" << endl;
		}
	}
	else if (Descr->f_BLT_database_embedded) {

		knowledge_base K;

		if (f_v) {
			cout << "geometric_object_create::init before K.retrieve_BLT_set_from_database_embedded" << endl;
		}
		K.retrieve_BLT_set_from_database_embedded(F,
				Descr->BLT_database_embedded_k,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after K.retrieve_BLT_set_from_database_embedded" << endl;
		}
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
	else if (Descr->f_elliptic_quadric_ovoid) {
		if (f_v) {
			cout << "geometric_object_create::init before create_elliptic_quadric_ovoid" << endl;
		}
		create_elliptic_quadric_ovoid(P,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after create_elliptic_quadric_ovoid" << endl;
		}
	}
	else if (Descr->f_ovoid_ST) {
		if (f_v) {
			cout << "geometric_object_create::init before create_ovoid_ST" << endl;
		}
		create_ovoid_ST(P,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after create_ovoid_ST" << endl;
		}
	}


	else if (Descr->f_Baer_substructure) {

		if (f_v) {
			cout << "geometric_object_create::init before create_Baer_substructure" << endl;
		}
		create_Baer_substructure(P,
			Pts, nb_pts,
			label_txt,
			label_tex,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after create_Baer_substructure" << endl;
		}

	}
	else if (Descr->f_orthogonal) {
		if (f_v) {
			cout << "geometric_object_create::init before F->create_orthogonal" << endl;
		}
		F->create_orthogonal(
				Descr->orthogonal_epsilon, P->n,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		// calls choose_anisotropic_form if necessary
		if (f_v) {
			cout << "geometric_object_create::init after F->create_orthogonal" << endl;
		}
	}
	else if (Descr->f_hermitian) {
		if (f_v) {
			cout << "geometric_object_create::init before F->create_hermitian" << endl;
		}
		F->create_hermitian(P->n,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		// creates a hermitian
		if (f_v) {
			cout << "geometric_object_create::init after F->create_hermitian" << endl;
		}
	}
	else if (Descr->f_cuspidal_cubic) {
		if (f_v) {
			cout << "geometric_object_create::init before create_cuspidal_cubic" << endl;
		}
		create_cuspidal_cubic(P,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after create_cuspidal_cubic" << endl;
		}
	}
	else if (Descr->f_twisted_cubic) {
		if (f_v) {
			cout << "geometric_object_create::init before create_twisted_cubic" << endl;
		}
		create_twisted_cubic(P,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after create_twisted_cubic" << endl;
		}
	}
	else if (Descr->f_elliptic_curve) {
		if (f_v) {
			cout << "geometric_object_create::init before create_elliptic_curve" << endl;
		}
		create_elliptic_curve(P,
				Descr->elliptic_curve_b, Descr->elliptic_curve_c,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after create_elliptic_curve" << endl;
		}
	}

#if 0
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
#endif
	else if (Descr->f_unital_XXq_YZq_ZYq) {

		if (f_v) {
			cout << "geometric_object_create::init before create_unital_XXq_YZq_ZYq" << endl;
		}

		create_unital_XXq_YZq_ZYq(P,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);

		if (f_v) {
			cout << "geometric_object_create::init after create_unital_XXq_YZq_ZYq" << endl;
		}
	}
#if 0
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
#endif

#if 0
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
#endif

	else if (Descr->f_whole_space) {
		if (f_v) {
			cout << "geometric_object_create::init before create_whole_space" << endl;
		}
		create_whole_space(P,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after create_whole_space" << endl;
		}
	}
	else if (Descr->f_hyperplane) {
		if (f_v) {
			cout << "geometric_object_create::init before create_hyperplane" << endl;
		}
		create_hyperplane(P,
				Descr->pt,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after create_hyperplane" << endl;
		}
	}
	else if (Descr->f_segre_variety) {
		if (f_v) {
			cout << "geometric_object_create::init before F->create_segre_variety" << endl;
		}
		F->create_segre_variety(
				Descr->segre_variety_a, Descr->segre_variety_b,
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after F->create_segre_variety" << endl;
		}
	}
	else if (Descr->f_Maruta_Hamada_arc) {
		if (f_v) {
			cout << "geometric_object_create::init before P->Arc_in_projective_space->create_Maruta_Hamada_arc" << endl;
		}
		P->Arc_in_projective_space->create_Maruta_Hamada_arc(
				label_txt,
				label_tex,
			nb_pts, Pts,
			verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after P->Arc_in_projective_space->create_Maruta_Hamada_arc" << endl;
		}
	}
	else if (Descr->f_projective_variety) {


		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = orbiter_kernel_system::Orbiter->get_object_of_type_polynomial_ring(Descr->projective_variety_ring_label);

		if (f_v) {
			cout << "geometric_object_create::init before HPD->create_projective_variety" << endl;
		}
		HPD->create_projective_variety(
				Descr->variety_label_txt,
				Descr->variety_label_tex,
				Descr->variety_coeffs,
				label_txt,
				label_tex,
				nb_pts, Pts,
				verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after HPD->create_projective_variety" << endl;
		}

	}
	else if (Descr->f_intersection_of_zariski_open_sets) {

		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = orbiter_kernel_system::Orbiter->get_object_of_type_polynomial_ring(Descr->intersection_of_zariski_open_sets_ring_label);

		if (f_v) {
			cout << "geometric_object_create::init before HPD->create_intersection_of_zariski_open_sets" << endl;
		}
		HPD->create_intersection_of_zariski_open_sets(
				Descr->variety_label_txt,
				Descr->variety_label_tex,
				Descr->Variety_coeffs,
				label_txt,
				label_tex,
				nb_pts, Pts,
				verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after HPD->create_intersection_of_zariski_open_sets" << endl;
		}
	}
	else if (Descr->f_number_of_conditions_satisfied) {

		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = orbiter_kernel_system::Orbiter->get_object_of_type_polynomial_ring(Descr->number_of_conditions_satisfied_ring_label);

		if (f_v) {
			cout << "geometric_object_create::init before HPD->number_of_conditions_satisfied" << endl;
		}
		HPD->number_of_conditions_satisfied(
				Descr->variety_label_txt,
				Descr->variety_label_tex,
				Descr->Variety_coeffs,
				Descr->number_of_conditions_satisfied_fname,
				label_txt,
				label_tex,
				nb_pts, Pts,
				verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after HPD->number_of_conditions_satisfied" << endl;
		}
	}



	else if (Descr->f_projective_curve) {

		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = orbiter_kernel_system::Orbiter->get_object_of_type_polynomial_ring(Descr->projective_curve_ring_label);

		if (f_v) {
			cout << "geometric_object_create::init before HPD->create_projective_curve" << endl;
		}
		HPD->create_projective_curve(
				Descr->curve_label_txt,
				Descr->curve_label_tex,
				Descr->curve_coeffs,
				label_txt,
				label_tex,
				nb_pts, Pts,
				verbose_level);
		if (f_v) {
			cout << "geometric_object_create::init after HPD->create_projective_curve" << endl;
		}
	}

	else if (Descr->f_set) {

		Lint_vec_scan(Descr->set_text, Pts, nb_pts);

	}
	else {
		cout << "geometric_object_create::init nothing to create" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "geometric_object_create::init created a set of size " << nb_pts << endl;
		Lint_vec_print_fully(cout, Pts, nb_pts);
		cout << endl;

		//lint_vec_print(cout, Pts, nb_pts);
		//cout << endl;
	}



	if (f_v) {
		cout << "geometric_object_create::init done" << endl;
	}
}

void geometric_object_create::create_elliptic_quadric_ovoid(
		projective_space *P,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_object_create::create_elliptic_quadric_ovoid" << endl;
	}
	int epsilon = -1;
	int c1 = 1, c2 = 0, c3 = 0;
	int i, j, d, h;
	int *v, *w;
	geometry_global Gg;

	if (P->n != 3) {
		cout << "geometric_object_create::create_elliptic_quadric_ovoid n != 3" << endl;
		exit(1);
	}
	d = P->n + 1;

	nb_pts = Gg.nb_pts_Qepsilon(epsilon, P->n, P->q);

	v = NEW_int(P->n + 1);
	w = NEW_int(P->n + 1);
	Pts = NEW_lint(P->N_points);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
	}
	P->F->Linear_algebra->choose_anisotropic_form(c1, c2, c3, verbose_level);
	for (i = 0; i < nb_pts; i++) {
		P->F->Orthogonal_indexing->Q_epsilon_unrank(v, 1, epsilon, P->n, c1, c2, c3, i, 0 /* verbose_level */);
		for (h = 0; h < d; h++) {
			w[h] = v[h];
		}
		j = P->rank_point(w);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
		}
	}

#if 0
	cout << "list of points on the ovoid:" << endl;
	cout << nb_pts << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];
	char str2[1000];

	snprintf(str, sizeof(str), "_q%d", P->q);
	snprintf(str2, sizeof(str2), "\\_q%d", P->q);


	label_txt.assign("ovoid");
	label_txt.append(str);
	label_tex.assign("ovoid");
	label_tex.append(str2);

	//write_set_to_file(fname, L, N, verbose_level);

	FREE_int(v);
	FREE_int(w);
	//FREE_int(L);
	if (f_v) {
		cout << "geometric_object_create::create_elliptic_quadric_ovoid done" << endl;
	}
}

void geometric_object_create::create_ovoid_ST(
		projective_space *P,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
// Suzuki Tits ovoid in PG(3,2^(2r+1)),
// following Heinz Lueneburg: Translation planes, 1980, Chapter IV
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_object_create::create_ovoid_ST" << endl;
	}
	int i, d, x, y, z, r, sigma, sigma_plus_two;
	long int a;
	int v[4];
	number_theory::number_theory_domain NT;

	if (EVEN(P->F->e)) {
		cout << "geometric_object_create::create_ovoid_ST need odd field degree" << endl;
		exit(1);
	}
	if (P->F->p != 2) {
		cout << "geometric_object_create::create_ovoid_ST F->p != 2" << endl;
		exit(1);
	}
	if (P->n != 3) {
		cout << "geometric_object_create::create_ovoid_ST need n == 3" << endl;
		exit(1);
	}

	r = (P->F->e - 1) >> 1;

	sigma = NT.i_power_j(2, r + 1);
	sigma_plus_two = sigma + 2;


	if (f_v) {
		cout << "geometric_object_create::create_ovoid_ST r = " << r << endl;
		cout << "geometric_object_create::create_ovoid_ST sigma = " << sigma << endl;
		cout << "geometric_object_create::create_ovoid_ST sigma_plus_two = " << sigma_plus_two << endl;
	}

	d = P->n + 1;

	nb_pts = P->F->q * P->F->q + 1;

	Pts = NEW_lint(nb_pts);

	i = 0;
	Pts[i++] = 1; // (0,1,0,0)
	for (x = 0; x < P->F->q; x++) {
		for (y = 0; y < P->F->q; y++) {

			z = P->F->add3(P->F->mult(x, y), P->F->power(x, sigma_plus_two), P->F->power(y, sigma));

			v[0] = 1;
			v[1] = z;
			v[2] = x;
			v[3] = y;

			a = P->rank_point(v);
			Pts[i++] = a;
		}
	}

	if (i != nb_pts) {
		cout << "geometric_object_create::create_ovoid_ST i != nb_pts" << endl;
	}
	if (f_v) {
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			cout << setw(4) << i << " : ";
			a = Pts[i];
			P->unrank_point(v, a);
			Int_vec_print(cout, v, d);
			cout << " : " << setw(5) << a << endl;
		}
	}

	char str[1000];
	char str2[1000];

	snprintf(str, sizeof(str), "_q%d", P->q);
	snprintf(str2, sizeof(str2), "\\_q%d", P->q);


	label_txt.assign("ovoid_ST");
	label_txt.append(str);
	label_tex.assign("ovoid\\_ST");
	label_tex.append(str2);

	if (f_v) {
		cout << "geometric_object_create::create_ovoid_ST done" << endl;
	}
}

void geometric_object_create::create_cuspidal_cubic(
		projective_space *P,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_object_create::create_cuspidal_cubic" << endl;
	}
	//int n = 2;
	long int i, a, d, s, t;
	int *v;
	int v2[2];

	if (P->n != 2) {
		cout << "geometric_object_create::create_cuspidal_cubic n != 2" << endl;
		exit(1);
	}
	d = P->n + 1;
	nb_pts = P->q + 1;

	v = NEW_int(d);
	Pts = NEW_lint(P->N_points);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		P->F->PG_element_unrank_modified(v2, 1, 2, i);
		s = v2[0];
		t = v2[1];
		v[0] = P->F->mult(P->F->power(s, 3), P->F->power(t, 0));
		v[1] = P->F->mult(P->F->power(s, 2), P->F->power(t, 1));
		v[2] = P->F->mult(P->F->power(s, 0), P->F->power(t, 3));
#if 0
		for (j = 0; j < d; j++) {
			v[j] = F->mult(F->power(s, n - j), F->power(t, j));
		}
#endif
		a = P->rank_point(v);
		Pts[i] = a;
		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, d);
			cout << " : " << setw(5) << a << endl;
		}
	}

#if 0
	cout << "list of points on the cubic:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];
	char str2[1000];
	snprintf(str, sizeof(str), "cuspidal_cubic_%d", P->q);
	snprintf(str2, sizeof(str2), "cuspidal\\_cubic\\_%d", P->q);
	label_txt.assign(str);
	label_tex.assign(str2);
	//write_set_to_file(fname, L, N, verbose_level);


#if 0
	long int nCk;
	combinatorics_domain Combi;
	int k = 6;
	int rk;
	int idx[6];
	int *subsets;

	nCk = Combi.int_n_choose_k(nb_pts, k);
	subsets = NEW_int(nCk * k);
	for (rk = 0; rk < nCk; rk++) {
		Combi.unrank_k_subset(rk, idx, nb_pts, k);
		for (i = 0; i < k; i++) {
			subsets[rk * k + i] = Pts[idx[i]];
		}
	}


	string fname2;

	snprintf(str, sizeof(str), "cuspidal_cubic_%d_subsets_%d.txt", q, k);
	fname2.assign(str);

	{

		ofstream fp(fname2);

		for (rk = 0; rk < nCk; rk++) {
			fp << k;
			for (i = 0; i < k; i++) {
				fp << " " << subsets[rk * k + i];
			}
			fp << endl;
		}
		fp << -1 << endl;

	}

	file_io Fio;

	cout << "Written file " << fname2 << " of size " << Fio.file_size(fname2) << endl;

#endif



	FREE_int(v);
	//FREE_int(L);
	if (f_v) {
		cout << "geometric_object_create::create_cuspidal_cubic done" << endl;
	}
}

void geometric_object_create::create_twisted_cubic(
		projective_space *P,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_object_create::create_twisted_cubic" << endl;
	}
	//int n = 3;
	long int i, j, d, s, t;
	int *v;
	int v2[2];

	if (P->n != 3) {
		cout << "geometric_object_create::create_twisted_cubic n != 3" << endl;
		exit(1);
	}
	d = P->n + 1;

	nb_pts = P->q + 1;

	v = NEW_int(P->n + 1);
	Pts = NEW_lint(P->N_points);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		P->F->PG_element_unrank_modified(v2, 1, 2, i);
		s = v2[0];
		t = v2[1];
		v[0] = P->F->mult(P->F->power(s, 3), P->F->power(t, 0));
		v[1] = P->F->mult(P->F->power(s, 2), P->F->power(t, 1));
		v[2] = P->F->mult(P->F->power(s, 1), P->F->power(t, 2));
		v[3] = P->F->mult(P->F->power(s, 0), P->F->power(t, 3));
		j = P->rank_point(v);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
		}
	}

#if 0
	cout << "list of points on the twisted cubic:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];
	char str2[1000];
	snprintf(str, sizeof(str), "twisted_cubic_%d", P->q);
	snprintf(str2, sizeof(str2), "twisted\\_cubic\\_%d", P->q);
	label_txt.assign(str);
	label_tex.assign(str2);
	//write_set_to_file(fname, L, N, verbose_level);

	FREE_int(v);
	//FREE_int(L);
	if (f_v) {
		cout << "geometric_object_create::create_twisted_cubic done" << endl;
	}
}

void geometric_object_create::create_elliptic_curve(
		projective_space *P,
	int elliptic_curve_b, int elliptic_curve_c,
	std::string &label_txt,
	std::string &label_tex,
	int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_object_create::create_elliptic_curve" << endl;
	}
	//int n = 2;
	long int i, a, d;
	int *v;
	number_theory::elliptic_curve *E;

	if (P->n != 2) {
		cout << "geometric_object_create::create_elliptic_curve n != 2" << endl;
		exit(1);
	}
	d = P->n + 1;

	nb_pts = P->q + 1;

	E = NEW_OBJECT(number_theory::elliptic_curve);
	v = NEW_int(P->n + 1);
	Pts = NEW_lint(P->N_points);

	E->init(P->F, elliptic_curve_b, elliptic_curve_c,
			verbose_level);

	nb_pts = E->nb;

	if (f_v) {
		cout << "i : point : projective rank" << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		P->F->PG_element_rank_modified_lint(E->T + i * d, 1, d, a);
		Pts[i] = a;
		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, E->T + i * d, d);
			cout << " : " << setw(5) << a << endl;
		}
	}

#if 0
	cout << "list of points on the elliptic curve:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];
	char str2[1000];
	snprintf(str, sizeof(str), "elliptic_curve_b%d_c%d_q%d",
			elliptic_curve_b, elliptic_curve_c, P->q);
	snprintf(str2, sizeof(str2), "elliptic\\_curve\\_b%d\\_c%d\\_q%d",
			elliptic_curve_b, elliptic_curve_c, P->q);
	label_txt.assign(str);
	label_tex.assign(str2);
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_OBJECT(E);
	FREE_int(v);
	//FREE_int(L);
	if (f_v) {
		cout << "geometric_object_create::create_elliptic_curve done" << endl;
	}
}

void geometric_object_create::create_unital_XXq_YZq_ZYq(
		projective_space *P,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_object_create::create_unital_XXq_YZq_ZYq" << endl;
	}
	//int n = 2;
	if (P->n != 2) {
		cout << "geometric_object_create::create_unital_XXq_YZq_ZYq n != 2" << endl;
		exit(1);
	}
	int i, rk, d;
	int *v;

	d = P->n + 1;

	v = NEW_int(d);
	Pts = NEW_lint(P->N_points);


	create_unital_XXq_YZq_ZYq_brute_force(P, Pts, nb_pts, verbose_level - 1);


	if (f_v) {
		cout << "i : point : projective rank" << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		rk = Pts[i];
		P->unrank_point(v, rk);
		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, d);
			cout << " : " << setw(5) << rk << endl;
		}
	}


	char str[1000];
	char str2[1000];
	snprintf(str, sizeof(str), "unital_XXq_YZq_ZYq_Q%d", P->q);
	snprintf(str2, sizeof(str2), "unital\\_XXq\\_YZq\\_ZYq\\_Q%d", P->q);
	label_txt.assign(str);
	label_tex.assign(str2);

	FREE_int(v);
	if (f_v) {
		cout << "geometric_object_create::create_unital_XXq_YZq_ZYq done" << endl;
	}
}

void geometric_object_create::create_whole_space(
		projective_space *P,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i; //, d;

	if (f_v) {
		cout << "geometric_object_create::create_whole_space" << endl;
	}
	//d = n + 1;

	Pts = NEW_lint(P->N_points);
	nb_pts = P->N_points;
	for (i = 0; i < P->N_points; i++) {
		Pts[i] = i;
	}

	char str[1000];
	char str2[1000];
	snprintf(str, sizeof(str), "whole_space_PG_%d_%d", P->n, P->q);
	snprintf(str2, sizeof(str2), "whole\\_space\\_PG\\_%d\\_%d", P->n, P->q);
	label_txt.assign(str);
	label_tex.assign(str2);

	if (f_v) {
		cout << "geometric_object_create::create_whole_space done" << endl;
	}
}

void geometric_object_create::create_hyperplane(
		projective_space *P,
	int pt,
	std::string &label_txt,
	std::string &label_tex,
	int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, d, a;
	int *v1;
	int *v2;

	if (f_v) {
		cout << "geometric_object_create::create_hyperplane pt=" << pt << endl;
	}
	d = P->n + 1;
	v1 = NEW_int(d);
	v2 = NEW_int(d);

	P->unrank_point(v1, pt);
	Pts = NEW_lint(P->N_points);
	nb_pts = 0;
	for (i = 0; i < P->N_points; i++) {
		P->unrank_point(v2, i);
		a = P->F->Linear_algebra->dot_product(d, v1, v2);
		if (a == 0) {
			Pts[nb_pts++] = i;
			if (f_v) {
				cout << setw(4) << nb_pts - 1 << " : ";
				Int_vec_print(cout, v2, d);
				cout << " : " << setw(5) << i << endl;
			}
		}
	}

	char str[1000];
	char str2[1000];
	snprintf(str, sizeof(str), "hyperplane_PG_%d_%d_pt%d", P->n, P->q, pt);
	snprintf(str2, sizeof(str2), "hyperplane\\_PG\\_%d\\_%d\\_pt%d", P->n, P->q, pt);
	label_txt.assign(str);
	label_tex.assign(str2);

	FREE_int(v1);
	FREE_int(v2);
	if (f_v) {
		cout << "geometric_object_create::create_hyperplane done" << endl;
	}
}

void geometric_object_create::create_Baer_substructure(
		projective_space *P,
	long int *&Pts, int &nb_pts,
	std::string &label_txt,
	std::string &label_tex,
	int verbose_level)
// assumes we are in PG(n,Q) where Q = q^2
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_object_create::create_Baer_substructure" << endl;
	}

	// projective space over the big field FQ = this

	number_theory::number_theory_domain NT;
	int Q = P->q;
	int q = NT.i_power_j(P->F->p, P->F->e >> 1);
	if (f_v) {
		cout << "geometric_object_create::create_Baer_substructure Q=" << Q << " q=" << q << endl;
	}

	int sz;
	int *v;
	int d = P->n + 1;
	int i, j, a, b, index, f_is_in_subfield;

	if (f_v) {
		cout << "geometric_object_create::create_Baer_substructure Q=" << Q << endl;
		cout << "geometric_object_create::create_Baer_substructure q=" << q << endl;
	}

	index = (Q - 1) / (q - 1);

	if (f_v) {
		cout << "geometric_object_create::create_Baer_substructure index=" << index << endl;
	}

	v = NEW_int(d);
	Pts = NEW_lint(P->N_points);

	sz = 0;
	for (i = 0; i < P->N_points; i++) {

		P->F->PG_element_unrank_modified(v, 1, d, i);

		for (j = 0; j < d; j++) {
			a = v[j];
			b = P->F->log_alpha(a);
			f_is_in_subfield = FALSE;

			if (a == 0 || (b % index) == 0) {
				f_is_in_subfield = TRUE;
			}
			if (!f_is_in_subfield) {
				break;
			}
		}
		if (j == d) {
			Pts[nb_pts++] = i;
		}
	}
	cout << "the Baer substructure "
			"PG(" << P->n << "," << q << ") inside "
			"PG(" << P->n << "," << Q << ") has size "
			<< sz << ":" << endl;
	for (i = 0; i < sz; i++) {
		cout << Pts[i] << " ";
	}
	cout << endl;



	char str[1000];

	//write_set_to_file(fname, S, sz, verbose_level);


	snprintf(str, sizeof(str), "PG_%d_%d", P->n, P->F->q);
	label_txt.assign("Baer_substructure_");
	label_txt.append(str);

	snprintf(str, sizeof(str), "PG(%d,%d)", P->n, P->F->q);
	label_tex.assign("Baer\\_substructure\\_");
	label_tex.append(str);

	FREE_int(v);
	//FREE_int(S);
	if (f_v) {
		cout << "geometric_object_create::create_Baer_substructure done" << endl;
	}
}

void geometric_object_create::create_unital_XXq_YZq_ZYq_brute_force(
		projective_space *P,
		long int *U, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int *v;
	long int e, i, a;
	long int X, Y, Z, Xq, Yq, Zq;

	if (f_v) {
		cout << "geometric_object_create::create_unital_XXq_YZq_ZYq" << endl;
	}
	if (P->n != 2) {
		cout << "geometric_object_create::create_unital_XXq_YZq_ZYq "
				"n != 2" << endl;
		exit(1);
 	}
	if (ODD(P->F->e)) {
		cout << "geometric_object_create::create_unital_XXq_YZq_ZYq "
				"ODD(F->e)" << endl;
		exit(1);
 	}

	v = NEW_int(3);
	e = P->F->e >> 1;
	if (f_vv) {
		cout << "e=" << e << endl;
	}
	sz = 0;
	for (i = 0; i < P->N_points; i++) {
		P->unrank_point(v, i);
		if (f_vvv) {
			cout << "i=" << i << " : ";
			Int_vec_print(cout, v, 3);
			//cout << endl;
		}
		X = v[0];
		Y = v[1];
		Z = v[2];
		Xq = P->F->frobenius_power(X, e);
		Yq = P->F->frobenius_power(Y, e);
		Zq = P->F->frobenius_power(Z, e);
		a = P->F->add3(P->F->mult(X, Xq), P->F->mult(Y, Zq), P->F->mult(Z, Yq));
		if (f_vvv) {
			cout << " a=" << a << endl;
		}
		if (a == 0) {
			//cout << "a=0, adding i=" << i << endl;
			U[sz++] = i;
			//int_vec_print(cout, U, sz);
			//cout << endl;
		}
	}
	if (f_vv) {
		cout << "we found " << sz << " points:" << endl;
		Lint_vec_print(cout, U, sz);
		cout << endl;
		P->print_set(U, sz);
	}
	FREE_int(v);

	if (f_v) {
		cout << "geometric_object_create::create_unital_XXq_YZq_ZYq "
				"done" << endl;
	}
}



}}}



