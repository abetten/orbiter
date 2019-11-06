/*
 * exceptional_isomorphism_O4_main.cpp
 *
 *  Created on: Mar 31, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

using namespace orbiter;

// global data:

int t0; // the system time when the program started

int main(int argc, char **argv);
void do_it(int q, int verbose_level);
void do_2to4(int q, int nb_gens,
		const char **gens1, const char **gens2, int *f_switch, int verbose_level);
void do_4to5(int q, int nb_gens,
		const char **gens,
		int verbose_level);


int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_q = FALSE;
	int q = 0;
	int f_2to4 = FALSE;
	int nb_gens = 0;
	const char *gens1[1000];
	const char *gens2[1000];
	int f_switch[1000];
	int f_4to5 = FALSE;
	os_interface Os;

	t0 = Os.os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-2to4") == 0) {
			f_2to4 = TRUE;
			nb_gens = atoi(argv[++i]);
			for (int j = 0; j < nb_gens; j++) {
				gens1[j] = argv[++i];
				gens2[j] = argv[++i];
				f_switch[j] = atoi(argv[++i]);
			}
			cout << "-2to4 " << nb_gens;
			for (int j = 0; j < nb_gens; j++) {
				cout << " " << gens1[j] << " " << gens2[j] << " " << f_switch[j];
			}
			cout << endl;
			}
		else if (strcmp(argv[i], "-4to5") == 0) {
			f_4to5 = TRUE;
			nb_gens = atoi(argv[++i]);
			for (int j = 0; j < nb_gens; j++) {
				gens1[j] = argv[++i];
			}
			cout << "-4to5 " << nb_gens;
			for (int j = 0; j < nb_gens; j++) {
				cout << " " << gens1[j];
			}
			cout << endl;
			}
		}
	if (!f_q) {
		cout << "please use -q <q>" << endl;
		exit(1);
		}

	if (f_2to4) {
		do_2to4(q, nb_gens, gens1, gens2, f_switch, verbose_level);
	}
	else if (f_4to5) {
		do_4to5(q, nb_gens, gens1, verbose_level);
	}
	else {
		do_it(q, verbose_level);
	}
	the_end_quietly(t0);
}

void do_it(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	action *A5;
	action *A4;
	action *A2;
	int f_semilinear = FALSE;
	int f_basis = TRUE;
	int p, h, i, j;
	number_theory_domain NT;

	if (f_v) {
		cout << "do_it q=" << q << endl;
	}
	A5 = NEW_OBJECT(action);
	A4 = NEW_OBJECT(action);
	A2 = NEW_OBJECT(action);
	NT.is_prime_power(q, p, h);
	if (h > 1)
		f_semilinear = TRUE;
	else
		f_semilinear = FALSE;


	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

	if (f_v) {
		cout << "do_it before A5->init_orthogonal_group" << endl;
	}
	A5->init_orthogonal_group(0 /*epsilon*/, 5, F,
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);

	if (f_v) {
		cout << "do_it before A4->init_orthogonal_group" << endl;
	}
	A4->init_orthogonal_group(1 /*epsilon*/, 4, F,
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);


	vector_ge *nice_gens;
	nice_gens = NEW_OBJECT(vector_ge);

	if (f_v) {
		cout << "do_it before A2->init_projective_group" << endl;
	}
	A2->init_projective_group(2, F, f_semilinear,
		TRUE /*f_basis*/,
		nice_gens,
		verbose_level);


	if (!A5->f_has_strong_generators) {
		cout << "action A5 does not have strong generators" << endl;
		exit(1);
		}

	exceptional_isomorphism_O4 *Iso;

	Iso = NEW_OBJECT(exceptional_isomorphism_O4);

	if (f_v) {
		cout << "do_it before Iso->init" << endl;
	}
	Iso->init(F, A2, A4, A5, verbose_level);


	sims *G;

	G = A4->Sims;
	cout << "checking all group elements in O^+(4,q):" << endl;

	sims *G5;

	G5 = A5->Sims;

	longinteger_object go;
	int goi;
	longinteger_object go5;
	int go5i;
	int f_switch;
	int *E5;
	int *E4a;
	int *E4b;
	int *E2a;
	int *E2b;

	E5 = NEW_int(A5->elt_size_in_int);
	E4a = NEW_int(A4->elt_size_in_int);
	E4b = NEW_int(A4->elt_size_in_int);
	E2a = NEW_int(A2->elt_size_in_int);
	E2b = NEW_int(A2->elt_size_in_int);

	G->group_order(go);
	goi = go.as_int();
	G5->group_order(go5);
	go5i = go5.as_int();

	cout << "group order O^+(4," << q << ") is " << go << " as int " << goi << endl;
	cout << "group order O(5," << q << ") is " << go5 << " as int " << go5i << endl;


	if (goi < 1000) {
		for (i = 0; i < goi; i++) {
			cout << "i=" << i << " / " << goi << endl;
			G->element_unrank_lint(i, E4a);

			cout << "E4a=" << endl;
			A4->element_print_quick(E4a, cout);


			Iso->apply_4_to_2(E4a, f_switch, E2a, E2b, 0 /*verbose_level*/);
			cout << "after isomorphism:" << endl;
			cout << "f_switch=" << f_switch << endl;
			cout << "E2a=" << endl;
			A2->element_print_quick(E2a, cout);
			cout << "E2b=" << endl;
			A2->element_print_quick(E2b, cout);

			Iso->apply_2_to_4(f_switch, E2a, E2b, E4b, 0 /*verbose_level*/);
			j = G->element_rank_lint(E4b);
			cout << "rank returns j=" << j << endl;
			if (j != i) {
				cout << "j != i" << endl;
				cout << "i=" << i << endl;
				cout << "j=" << j << endl;
				exit(1);
			}
#if 0
			Iso->apply_4_to_5(E4a, E5, 0 /*verbose_level*/);
			cout << "E5=" << endl;
			A5->element_print_quick(E5, cout);
			j = G5->element_rank_int(E5);
			cout << "rank in O(5,q)=" << j << endl;
#endif
		}
	}
	if (q == 71 || q == 23) {
		int data23[] = {
				// 3 generators for Sym(4) in PSL(2,23):
				1,9,20,22,
				1,7,2,22,
				1,19,1,22
		};
		int data71[] = {
				// 3 generators for Sym(4) in PSL(2,71):
#if 0
				1,1,64,70,
				1,5,38,17,
				1,29,10,70
#else
				1,29,10,70,
				1,43,3,70,
				1,7,22,70
#endif
		};
		for (i = 0; i < 3; i++) {
			if (q == 23) {
				A2->make_element(E2a, data23 + i * 4, verbose_level);
			}
			else if (q == 71) {
				A2->make_element(E2a, data71 + i * 4, verbose_level);
			}
			else {
				cout << "unknown value of q" << endl;
				exit(1);
			}
			A2->element_one(E2b, 0 /* verbose_level */);
			f_switch = FALSE;
			Iso->apply_2_to_4(f_switch, E2a, E2b, E4a, verbose_level);

			cout << "E4a=" << endl;
			A4->element_print_quick(E4a, cout);
			//A4->element_print_quick(E4a, cout);
			A4->print_for_make_element(cout, E4a);
			cout << endl;

			j = G->element_rank_lint(E4a);
			cout << "rank =" << j << endl;

#if 1
			Iso->apply_4_to_5(E4a, E5, verbose_level);
			cout << "E5=" << endl;
			A5->element_print_quick(E5, cout);
			A5->print_for_make_element(cout, E5);
			cout << endl;
			j = G5->element_rank_lint(E5);
			cout << "rank in O(5,q)=" << j << endl;
#endif
		}
	}

}


void do_2to4(int q, int nb_gens,
		const char **gens1, const char **gens2, int *f_switch,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	action *A4;
	action *A2;
	int f_semilinear = FALSE;
	int f_basis = TRUE;
	int p, h, i, j;
	number_theory_domain NT;

	if (f_v) {
		cout << "do_2to4 q=" << q << endl;
	}
	A4 = NEW_OBJECT(action);
	A2 = NEW_OBJECT(action);
	NT.is_prime_power(q, p, h);
	if (h > 1)
		f_semilinear = TRUE;
	else
		f_semilinear = FALSE;


	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

	if (f_v) {
		cout << "do_2to4 before A4->init_orthogonal_group" << endl;
	}
	A4->init_orthogonal_group(1 /*epsilon*/, 4, F,
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);


	vector_ge *nice_gens;
	nice_gens = NEW_OBJECT(vector_ge);

	if (f_v) {
		cout << "do_2to4 before A2->init_projective_group" << endl;
	}
	A2->init_projective_group(2, F, f_semilinear,
		TRUE /*f_basis*/,
		nice_gens,
		verbose_level);



	exceptional_isomorphism_O4 *Iso;

	Iso = NEW_OBJECT(exceptional_isomorphism_O4);

	if (f_v) {
		cout << "do_2to4 before Iso->init" << endl;
	}
	Iso->init(F, A2, A4, NULL, verbose_level);


	sims *G;

	G = A4->Sims;

	longinteger_object go;
	int goi;
	//int f_switch;
	int *E4a;
	int *E4b;
	int *E2a;
	int *E2b;

	E4a = NEW_int(A4->elt_size_in_int);
	E4b = NEW_int(A4->elt_size_in_int);
	E2a = NEW_int(A2->elt_size_in_int);
	E2b = NEW_int(A2->elt_size_in_int);

	G->group_order(go);
	goi = go.as_int();

	cout << "group order O^+(4," << q << ") is " << go << " as int " << goi << endl;


	for (i = 0; i < nb_gens; i++) {
		int *data1;
		int data1_len;
		int *data2;
		int data2_len;

		int_vec_scan(gens1[i], data1, data1_len);
		int_vec_scan(gens2[i], data2, data2_len);

		A2->make_element(E2a, data1, verbose_level);
		A2->make_element(E2b, data2, verbose_level);
		//A2->element_one(E2b, 0 /* verbose_level */);
		Iso->apply_2_to_4(f_switch[i], E2a, E2b, E4a, verbose_level);

		cout << "E4a=" << endl;
		A4->element_print_quick(E4a, cout);
		A4->element_print_latex(E4a, cout);
		//A4->element_print_quick(E4a, cout);
		A4->print_for_make_element(cout, E4a);
		cout << endl;

		j = G->element_rank_lint(E4a);
		cout << "rank =" << j << endl;

	}

}


void do_4to5(int q, int nb_gens,
		const char **gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	finite_field *F;
	action *A4;
	action *A5;
	action *A4_linear;
	action *A4_linear_on_lines;
	int f_semilinear = FALSE;
	int f_basis = TRUE;
	int p, h;
	number_theory_domain NT;

	if (f_v) {
		cout << "do_4to5 q=" << q << endl;
	}
	A4 = NEW_OBJECT(action);
	A5 = NEW_OBJECT(action);
	NT.is_prime_power(q, p, h);
	if (h > 1)
		f_semilinear = TRUE;
	else
		f_semilinear = FALSE;


	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

	if (f_v) {
		cout << "do_4to5 before A4->init_orthogonal_group" << endl;
	}
	A4->init_orthogonal_group(1 /*epsilon*/, 4, F,
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);


	vector_ge *nice_gens;
	nice_gens = NEW_OBJECT(vector_ge);

	if (f_v) {
		cout << "do_4to5 before A2->init_projective_group" << endl;
	}
	A5->init_orthogonal_group(0 /*epsilon*/, 5, F,
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);

	orthogonal *O4;
	orthogonal *O5;

	O4 = A4->G.AO->O;
	O5 = A5->G.AO->O;

	A4_linear = A4->subaction;
	A4_linear_on_lines = A4_linear->induced_action_on_grassmannian(2,
			verbose_level);

	sims *G;

	G = A4->Sims;


	sims *G5;

	G5 = A5->Sims;

	longinteger_object go;
	int goi;
	longinteger_object go5;
	int go5i;
	int *E5;
	int *E4;

	E5 = NEW_int(A5->elt_size_in_int);
	E4 = NEW_int(A4->elt_size_in_int);

	G->group_order(go);
	goi = go.as_int();
	G5->group_order(go5);
	go5i = go5.as_int();

	cout << "group order O^+(4," << q << ") is " << go << " as int " << goi << endl;
	cout << "group order O(5," << q << ") is " << go5 << " as int " << go5i << endl;

	W3q *w3q;

	w3q = NEW_OBJECT(W3q);

	if (f_v) {
		cout << "do_4to5 before w3q->init" << endl;
	}
	w3q->init(F, verbose_level);
	if (f_v) {
		cout << "do_4to5 after w3q->init" << endl;
	}

	//w3q->print_by_lines();

	w3q->print_by_points();

	int i, j;
	int Gram4[16];
	int Gram4_transformed[16];


	for (i = 0; i < nb_gens; i++) {
		int *data;
		int data_len;

		int_vec_scan(gens[i], data, data_len);

		A4->make_element(E4, data, verbose_level);


		int_vec_zero(Gram4, 16);
		Gram4[0 * 4 + 1] = 1;
		Gram4[1 * 4 + 0] = 1;
		Gram4[2 * 4 + 3] = 1;
		Gram4[3 * 4 + 2] = 1;

		if (f_v) {
			cout << "Gram4 matrix:" << endl;
			print_integer_matrix_width(cout, Gram4, 4, 4, 4, 3);
			cout << "E4:" << endl;
			print_integer_matrix_width(cout, E4, 4, 4, 4, 3);
			}


		F->transform_form_matrix(E4, Gram4, Gram4_transformed, 4);
		// computes Gram_transformed = A * Gram * A^\top


		if (f_v) {
			cout << "Gram4_transformed:" << endl;
			print_integer_matrix_width(cout, Gram4_transformed, 4, 4, 4, 3);
			}

		int Basis[25];
		int Basis2[25];
		int rk_pt[5];
		int rk_line[5];
		int rk_line_image[5];
		int rk_pt_image[5];
		int idx;

		int_vec_zero(Basis, 25);
		Basis[0] = 1;
		Basis[1] = F->negate(1);
		Basis[2] = 1;
		Basis[5 + 1] = 1;
		Basis[10 + 2] = 1;
		Basis[15 + 3] = 1;
		Basis[20 + 4] = 1;

		cout << "input:" << endl;
		int_matrix_print(Basis, 5, 5);

		for (j = 0; j < 5; j++) {
			cout << "j=" << j << endl;
			rk_pt[j] = O5->rank_point(Basis + j * 5, 1, 0);
			cout << "rk_pt[j]=" << rk_pt[j] << endl;
			rk_line[j] = w3q->Lines[w3q->Line_idx[rk_pt[j]]];
			cout << "rk_line[j]=" << rk_line[j] << endl;
			rk_line_image[j] = A4_linear_on_lines->element_image_of(
					rk_line[j], E4, 0 /*verbose_level*/);
			cout << "rk_line_image[j]=" << rk_line_image[j] << endl;

			//int element_image_of(int a, void *elt, int verbose_level);

			idx = w3q->find_line(rk_line_image[j]);
			cout << "idx=" << idx << endl;

			rk_pt_image[j] = w3q->Q4_rk[idx];
			cout << "rk_pt_image[j]=" << rk_pt_image[j] << endl;
			O5->unrank_point(Basis2 + j * 5, 1, rk_pt_image[j], 0);
		}

		cout << "output:" << endl;
		int_matrix_print(Basis2, 5, 5);
	}


}
