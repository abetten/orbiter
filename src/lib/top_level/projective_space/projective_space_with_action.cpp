// projective_space_with_action.cpp
// 
// Anton Betten
//
// December 22, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



projective_space_with_action::projective_space_with_action()
{
	null();
}

projective_space_with_action::~projective_space_with_action()
{
	freeself();
}

void projective_space_with_action::null()
{
	q = 0;
	F = NULL;
	P = NULL;
	PA2 = NULL;
	Dom = NULL;
	QCDA = NULL;
	A = NULL;
	A_on_lines = NULL;
	Elt1 = NULL;
}

void projective_space_with_action::freeself()
{
	if (P) {
		FREE_OBJECT(P);
	}
	if (PA2) {
		FREE_OBJECT(PA2);
	}
	if (Dom) {
		FREE_OBJECT(Dom);
	}
	if (QCDA) {
		FREE_OBJECT(QCDA);
	}
	if (A) {
		FREE_OBJECT(A);
	}
	if (A_on_lines) {
		FREE_OBJECT(A_on_lines);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	null();
}

void projective_space_with_action::init(
	finite_field *F, int n, int f_semilinear,
	int f_init_incidence_structure,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::init" << endl;
	}
	projective_space_with_action::f_init_incidence_structure = f_init_incidence_structure;
	projective_space_with_action::n = n;
	projective_space_with_action::F = F;
	projective_space_with_action::f_semilinear = f_semilinear;
	d = n + 1;
	q = F->q;
	
	P = NEW_OBJECT(projective_space);
	P->init(n, F, 
		f_init_incidence_structure, 
		verbose_level);
	
	init_group(f_semilinear, verbose_level);

	if (n == 2) {
		Dom = NEW_OBJECT(quartic_curve_domain);

		if (f_v) {
			cout << "projective_space_with_action::init before Dom->init" << endl;
		}
		Dom->init(F, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init after Dom->init" << endl;
		}
		QCDA = NEW_OBJECT(quartic_curve_domain_with_action);
		if (f_v) {
			cout << "projective_space_with_action::init before QCDA->init" << endl;
		}
		QCDA->init(Dom, this, verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::init after QCDA->init" << endl;
		}
	}

	if (n >= 3) {
		if (f_v) {
			cout << "projective_space_with_action::init n >= 3, so we initialize a plane" << endl;
		}
		PA2 = NEW_OBJECT(projective_space_with_action);
		if (f_v) {
			cout << "projective_space_with_action::init before PA2->init" << endl;
		}
		PA2->init(F, 2, f_semilinear,
			f_init_incidence_structure,
			verbose_level - 2);
		if (f_v) {
			cout << "projective_space_with_action::init after PA2->init" << endl;
		}
	}


	
	Elt1 = NEW_int(A->elt_size_in_int);


	if (f_v) {
		cout << "projective_space_with_action::init done" << endl;
	}
}

void projective_space_with_action::init_group(
		int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "projective_space_with_action::init_group" << endl;
	}
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating linear group" << endl;
	}

	vector_ge *nice_gens;

	A = NEW_OBJECT(action);
	A->init_linear_group(
		F, d, 
		TRUE /*f_projective*/,
		FALSE /* f_general*/,
		FALSE /* f_affine */,
		f_semilinear,
		FALSE /* f_special */,
		nice_gens,
		0 /* verbose_level*/);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating linear group done" << endl;
	}
#if 0
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"before create_sims" << endl;
	}
	S = A->Strong_gens->create_sims(verbose_level - 2);

	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"after create_sims" << endl;
	}
#endif
	FREE_OBJECT(nice_gens);


	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating action on lines" << endl;
	}
	A_on_lines = A->induced_action_on_grassmannian(2, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::init_group "
				"creating action on lines done" << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::init_group done" << endl;
	}
}


void projective_space_with_action::canonical_form(
		projective_space_object_classifier_description *Canonical_form_PG_Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space_object_classifier *OC;

	if (f_v) {
		cout << "projective_space_with_action::canonical_form" << endl;
	}

	OC = NEW_OBJECT(projective_space_object_classifier);

	if (f_v) {
		cout << "projective_space_with_action::canonical_form before OC->do_the_work" << endl;
	}
	OC->do_the_work(
			Canonical_form_PG_Descr,
			this,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form after OC->do_the_work" << endl;
	}

	FREE_OBJECT(OC);

	if (f_v) {
		cout << "projective_space_with_action::canonical_form done" << endl;
	}
}

void projective_space_with_action::canonical_labeling(
	object_in_projective_space *OiP,
	int *canonical_labeling,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int *Incma;
	int *partition;
	int nb_rows, nb_cols;
	int *Aut, Aut_counter;
	int *Base, Base_length;
	int *Transversal_length;
	longinteger_object Ago;
	int N, i;
	nauty_interface Nau;


	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling"
				<< endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"before OiP->encode_incma" << endl;
	}
	OiP->encode_incma(Incma, nb_rows, nb_cols,
			partition, verbose_level - 1);
	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"after OiP->encode_incma" << endl;
	}
	if (verbose_level > 5) {
		cout << "projective_space_with_action::canonical_labeling "
				"Incma:" << endl;
		Orbiter->Int_vec.matrix_print_tight(Incma, nb_rows, nb_cols);
	}

	//canonical_labeling = NEW_int(nb_rows + nb_cols);
	for (i = 0; i < nb_rows + nb_cols; i++) {
		canonical_labeling[i] = i;
	}


	N = nb_rows + nb_cols;
	//L = nb_rows * nb_cols;

	if (f_vv) {
		cout << "projective_space_with_action::canonical_labeling "
				"initializing Aut, Base, "
				"Transversal_length" << endl;
	}
	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Transversal_length = NEW_int(N);

	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"calling nauty_interface_matrix_int" << endl;
	}


	int t0, t1, dt; //, tps;
	double delta_t_in_sec;
	os_interface Os;

	//tps = Os.os_ticks_per_second();
	t0 = Os.os_ticks();

	Nau.nauty_interface_matrix_int(
		Incma, nb_rows, nb_cols,
		canonical_labeling, partition,
		Aut, Aut_counter,
		Base, Base_length,
		Transversal_length, Ago, verbose_level - 3);

	t1 = Os.os_ticks();
	dt = t1 - t0;
	delta_t_in_sec = (double) t1 / (double) dt;

	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"done with nauty_interface_matrix_int, "
				"Ago=" << Ago << " dt=" << dt
				<< " delta_t_in_sec=" << delta_t_in_sec << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling "
				"done with nauty_interface_matrix_int, "
				"Ago=" << Ago << endl;
	}
	FREE_int(Aut);
	FREE_int(Base);
	FREE_int(Transversal_length);
	FREE_int(Incma);
	FREE_int(partition);
	if (f_v) {
		cout << "projective_space_with_action::canonical_labeling done"
				<< endl;
	}
}

strong_generators
*projective_space_with_action::set_stabilizer_of_object(
	object_in_projective_space *OiP, 
	int f_compute_canonical_form, bitvector *&Canonical_form,
	long int *canonical_labeling, int &canonical_labeling_len,
	int verbose_level)
// canonical_labeling[nb_rows + nb_cols] contains the canonical labeling
// where nb_rows and nb_cols is the encoding size ,
// which can be computed using
// object_in_projective_space::encoding_size(
//   int &nb_rows, int &nb_cols,
//   int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	action *A_linear;
	int *Incma;
	int *partition;
	//int *labeling;
	int nb_rows, nb_cols;
	int *Aut, Aut_counter;
	int *Base, Base_length;
	long int *Base_lint;
	int *Transversal_length;
	longinteger_object Ago;
	int N, i, j, a, L;
	combinatorics_domain Combi;
	file_io Fio;
	nauty_interface Nau;


	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_of_object" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	A_linear = A;

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"before OiP->encode_incma" << endl;
	}
	OiP->encode_incma(Incma, nb_rows, nb_cols, partition, verbose_level - 1);
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"after OiP->encode_incma" << endl;
	}
	if (verbose_level > 5) {
		cout << "projective_space_with_action::set_stabilizer_of_object Incma:" << endl;
		//int_matrix_print_tight(Incma, nb_rows, nb_cols);
	}

	//canonical_labeling = NEW_int(nb_rows + nb_cols);

	canonical_labeling_len = nb_rows + nb_cols;
	for (i = 0; i < canonical_labeling_len; i++) {
		canonical_labeling[i] = i;
	}

#if 0
	if (f_save_incma_in_and_out) {
		save_Levi_graph(save_incma_in_and_out_prefix,
				"Incma_in_%d_%d",
				Incma, nb_rows, nb_cols,
				canonical_labeling, canonical_labeling_len,
				verbose_level);

	}
#endif

	N = canonical_labeling_len;
	L = nb_rows * nb_cols;

	if (f_vv) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"initializing Aut, Base, Transversal_length" << endl;
	}
	Aut = NEW_int(N * N);
	Base = NEW_int(N);
	Base_lint = NEW_lint(N);
	Transversal_length = NEW_int(N);
	
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"calling Nau.nauty_interface_matrix_int" << endl;
	}
	int t0, t1, dt, tps;
	double delta_t_in_sec;
	os_interface Os;

	tps = Os.os_ticks_per_second();
	t0 = Os.os_ticks();

	int *can_labeling;

	can_labeling = NEW_int(canonical_labeling_len);

	Nau.nauty_interface_matrix_int(
		Incma, nb_rows, nb_cols,
		can_labeling, partition,
		Aut, Aut_counter, 
		Base, Base_length, 
		Transversal_length, Ago,
		verbose_level - 3);

	for (i = 0; i < canonical_labeling_len; i++) {
		canonical_labeling[i] = can_labeling[i];
	}
	FREE_int(can_labeling);

	Orbiter->Int_vec.copy_to_lint(Base, Base_lint, Base_length);

	t1 = Os.os_ticks();
	dt = t1 - t0;
	delta_t_in_sec = (double) dt / (double) tps;

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"done with Nau.nauty_interface_matrix_int, "
				"Ago=" << Ago << " dt=" << dt
				<< " delta_t_in_sec=" << delta_t_in_sec << endl;
	}
	if (verbose_level > 5) {
		int h;
		//int degree = nb_rows +  nb_cols;

		for (h = 0; h < Aut_counter; h++) {
			cout << "aut generator " << h << " / "
					<< Aut_counter << " : " << endl;
			//Combi.perm_print(cout, Aut + h * degree, degree);
			cout << endl;
		}
	}

	int *Incma_out;
	int ii, jj;
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"labeling:" << endl;
		//lint_vec_print(cout, canonical_labeling, canonical_labeling_len);
		cout << endl;
	}

	Incma_out = NEW_int(L);
	for (i = 0; i < nb_rows; i++) {
		ii = canonical_labeling[i];
		for (j = 0; j < nb_cols; j++) {
			jj = canonical_labeling[nb_rows + j] - nb_rows;
			//cout << "i=" << i << " j=" << j << " ii=" << ii
			//<< " jj=" << jj << endl;
			Incma_out[i * nb_cols + j] = Incma[ii * nb_cols + jj];
		}
	}
	if (f_vvv) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"Incma Out:" << endl;
		if (nb_rows < 20) {
			Orbiter->Int_vec.print_integer_matrix_width(cout,
					Incma_out, nb_rows, nb_cols, nb_cols, 1);
		}
		else {
			cout << "projective_space_with_action::set_stabilizer_of_object "
					"too large to print" << endl;
		}
	}



	if (f_compute_canonical_form) {
		
		Canonical_form = NEW_OBJECT(bitvector);
		Canonical_form->allocate(L);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out[i * nb_cols + j]) {
					a = i * nb_cols + j;
					//bitvector_set_bit(canonical_form, a);
					Canonical_form->m_i(a, 1);
				}
			}
		}

	}

#if 0
	if (f_save_incma_in_and_out) {
		save_Levi_graph(save_incma_in_and_out_prefix,
				"Incma_out_%d_%d",
				Incma_out, nb_rows, nb_cols,
				canonical_labeling, N,
				verbose_level);

	}
#endif

	FREE_int(Incma_out);


	nauty_interface_with_group Nauty;

	strong_generators *SG;

	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"before Nauty.reverse_engineer_linear_group_from_permutation_group" << endl;
		}
	Nauty.reverse_engineer_linear_group_from_permutation_group(
			A_linear,
			P,
			SG,
			N,
			Aut, Aut_counter,
			Base, Base_length,
			Base_lint,
			Transversal_length, Ago,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_of_object "
				"after Nauty.reverse_engineer_linear_group_from_permutation_group" << endl;
	}



	if (f_v) {
		cout << "before freeing Incma" << endl;
	}
	FREE_int(Incma);
	if (f_v) {
		cout << "before freeing partition" << endl;
	}
	FREE_int(partition);

	if (f_v) {
		cout << "before freeing Aut" << endl;
	}
	FREE_int(Aut);
	if (f_v) {
		cout << "before freeing Base" << endl;
	}
	FREE_int(Base);
	FREE_lint(Base_lint);
	if (f_v) {
		cout << "before freeing Transversal_length" << endl;
	}
	FREE_int(Transversal_length);



	if (f_v) {
		cout << "projective_space_with_action::set_stabilizer_of_object done" << endl;
	}
	return SG;
}


#if 0
void projective_space_with_action::save_Levi_graph(std::string &prefix,
		const char *mask,
		int *Incma, int nb_rows, int nb_cols,
		long int *canonical_labeling, int canonical_labeling_len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::save_Levi_graph" << endl;
	}
	file_io Fio;
	string fname_csv;
	string fname_bin;
	string fname_labeling;
	char str[1000];

	sprintf(str, mask, nb_rows, nb_cols);

	fname_csv.assign(prefix);
	fname_csv.append(str);
	fname_csv.append(".csv");

	fname_bin.assign(prefix);
	fname_bin.append(str);
	fname_bin.append(".graph");


	fname_labeling.assign(prefix);
	fname_labeling.append("_labeling");
	fname_labeling.append(".csv");

	latex_interface L;

#if 0
	cout << "labeling:" << endl;
	L.lint_vec_print_as_matrix(cout,
			canonical_labeling, N, 10 /* width */, TRUE /* f_tex */);
#endif

	Fio.lint_vec_write_csv(canonical_labeling, canonical_labeling_len,
			fname_labeling, "can_lab");
	Fio.int_matrix_write_csv(fname_csv, Incma, nb_rows, nb_cols);


	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);

	CG->create_Levi_graph_from_incidence_matrix(
			Incma, nb_rows, nb_cols,
			TRUE, canonical_labeling, verbose_level);
	CG->save(fname_bin, verbose_level);
	FREE_OBJECT(CG);
	if (f_v) {
		cout << "projective_space_with_action::save_Levi_graph done" << endl;
	}
}
#endif

void projective_space_with_action::report_fixed_points_lines_and_planes(
	int *Elt, ostream &ost, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_fixed_points_lines_and_planes" << endl;
	}

	if (P->n < 3) {
		cout << "projective_space_with_action::report_fixed_points_lines_and_planes P->n < 3" << endl;
		exit(1);
	}
	projective_space *P3;
	int i, j, cnt;
	int v[4];

	P3 = P;
	
	ost << "Fixed Objects:\\\\" << endl;



	ost << "The element" << endl;
	ost << "$$" << endl;
	A->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following fixed objects:\\\\" << endl;


	ost << "Fixed points:\\\\" << endl;

	cnt = 0;
	for (i = 0; i < P3->N_points; i++) {
		j = A->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
		}
	}

	ost << "There are " << cnt << " fixed points, they are: \\\\" << endl;
	for (i = 0; i < P3->N_points; i++) {
		j = A->element_image_of(i, Elt, 0 /* verbose_level */);
		F->PG_element_unrank_modified(v, 1, 4, i);
		if (j == i) {
			ost << i << " : ";
			Orbiter->Int_vec.print(ost, v, 4);
			ost << "\\\\" << endl;
			cnt++;
		}
	}

	ost << "Fixed Lines:\\\\" << endl;

	{
		action *A2;

		A2 = A->induced_action_on_grassmannian(2, 0 /* verbose_level*/);
	
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
				}
			}

		ost << "There are " << cnt << " fixed lines, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				ost << i << " : $\\left[";
				A2->G.AG->G->print_single_generator_matrix_tex(ost, i);
				ost << "\\right]$\\\\" << endl;
				cnt++;
				}
			}

		FREE_OBJECT(A2);
	}

	ost << "Fixed Planes:\\\\" << endl;

	{
		action *A2;

		A2 = A->induced_action_on_grassmannian(3, 0 /* verbose_level*/);
	
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				cnt++;
			}
		}

		ost << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
		cnt = 0;
		for (i = 0; i < A2->degree; i++) {
			j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
			if (j == i) {
				ost << i << " : $\\left[";
				A2->G.AG->G->print_single_generator_matrix_tex(ost, i);
				ost << "\\right]$\\\\" << endl;
				cnt++;
			}
		}

		FREE_OBJECT(A2);
	}

	if (f_v) {
		cout << "projective_space_with_action::report_fixed_points_lines_and_planes done" << endl;
	}
}

void projective_space_with_action::report_orbits_on_points_lines_and_planes(
	int *Elt, ostream &ost,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes" << endl;
	}

	if (P->n < 3) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes P->n < 3" << endl;
		exit(1);
	}
	//projective_space *P3;
	int order;

	longinteger_object full_group_order;
	order = A->element_order(Elt);

	full_group_order.create(order, __FILE__, __LINE__);

	//P3 = P;

	ost << "Fixed Objects:\\\\" << endl;



	ost << "The group generated by the element" << endl;
	ost << "$$" << endl;
	A->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following orbits:\\\\" << endl;

	ost << "Orbits on points:\\\\" << endl;


	schreier *Sch;

	Sch = NEW_OBJECT(schreier);
	A->all_point_orbits_from_single_generator(*Sch,
			Elt,
			0 /*verbose_level*/);
	Sch->print_orbit_lengths_tex(ost);


	FREE_OBJECT(Sch);

	ost << "Orbits on lines:\\\\" << endl;

	{
		action *A2;
		schreier *Sch;

		A2 = A->induced_action_on_grassmannian(2, 0 /* verbose_level*/);

		Sch = NEW_OBJECT(schreier);
		A2->all_point_orbits_from_single_generator(*Sch,
				Elt,
				0 /*verbose_level*/);
		Sch->print_orbit_lengths_tex(ost);


		FREE_OBJECT(Sch);
		FREE_OBJECT(A2);
	}

	ost << "Orbits on planes:\\\\" << endl;

	{
		action *A2;
		schreier *Sch;


		A2 = A->induced_action_on_grassmannian(3, 0 /* verbose_level*/);

		Sch = NEW_OBJECT(schreier);
		A2->all_point_orbits_from_single_generator(*Sch,
				Elt,
				0 /*verbose_level*/);
		Sch->print_orbit_lengths_tex(ost);


		FREE_OBJECT(Sch);
		FREE_OBJECT(A2);
	}

	if (f_v) {
		cout << "projective_space_with_action::report_orbits_on_points_lines_and_planes done" << endl;
	}
}

void projective_space_with_action::report_decomposition_by_single_automorphism(
	int *Elt, ostream &ost, std::string &fname_base,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism" << endl;
	}

#if 0
	if (P->n != 3) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism P->n != 3" << endl;
		exit(1);
	}
#endif
	//projective_space *P3;
	int order;

	longinteger_object full_group_order;
	order = A->element_order(Elt);

	full_group_order.create(order, __FILE__, __LINE__);

	//P3 = P;

	//ost << "Fixed Objects:\\\\" << endl;


#if 0
	ost << "The group generated by the element" << endl;
	ost << "$$" << endl;
	A->element_print_latex(Elt, ost);
	ost << "$$" << endl;
	ost << "has the following orbits:\\\\" << endl;

	ost << "Orbits on points:\\\\" << endl;
#endif

	schreier *Sch1;
	schreier *Sch2;
	incidence_structure *Inc;
	partitionstack *Stack;
	partitionstack S1;
	partitionstack S2;

	Sch1 = NEW_OBJECT(schreier);
	Sch2 = NEW_OBJECT(schreier);
	A->all_point_orbits_from_single_generator(*Sch1,
			Elt,
			0 /*verbose_level*/);

	//ost << "Orbits on lines:\\\\" << endl;

	Sch2 = NEW_OBJECT(schreier);
	A_on_lines->all_point_orbits_from_single_generator(*Sch2,
			Elt,
			0 /*verbose_level*/);
	//Sch->print_orbit_lengths_tex(ost);

	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism "
				"before incidence_and_stack_for_type_ij" << endl;
	}
	P->incidence_and_stack_for_type_ij(
		1 /* row_type */, 2 /* col_type */,
		Inc,
		Stack,
		0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism "
				"after incidence_and_stack_for_type_ij" << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism "
				"before S1.allocate" << endl;
	}
	S1.allocate(A->degree, 0 /* verbose_level */);
	S2.allocate(A_on_lines->degree, 0 /* verbose_level */);

	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism "
				"before Sch1->get_orbit_partition" << endl;
	}
	Sch1->get_orbit_partition(S1, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism "
				"before Sch2->get_orbit_partition" << endl;
	}
	Sch2->get_orbit_partition(S2, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism "
				"after Sch2->get_orbit_partition" << endl;
	}
	int i, j, sz;

	for (i = 1; i < S1.ht; i++) {
		if (f_v) {
			cout << "projective_space_with_action::report_decomposition_by_single_automorphism "
					"before Stack->split_cell (S1) i=" << i << endl;
		}
		Stack->split_cell(
				S1.pointList + S1.startCell[i],
				S1.cellSize[i], verbose_level);
	}
	int *set;
	set = NEW_int(A_on_lines->degree);
	for (i = 1; i < S2.ht; i++) {
		sz = S2.cellSize[i];
		Orbiter->Int_vec.copy(S2.pointList + S2.startCell[i], set, sz);
		for (j = 0; j < sz; j++) {
			set[j] += A->degree;
		}
		if (f_v) {
			cout << "projective_space_with_action::report_decomposition_by_single_automorphism "
					"before Stack->split_cell (S2) i=" << i << endl;
		}
		Stack->split_cell(set, sz, 0 /*verbose_level*/);
	}
	FREE_int(set);


	ost << "Considering the cyclic group generated by" << endl;
	ost << "$$" << endl;
	A->element_print_latex(Elt, ost);
	ost << "$$" << endl;

	if (Sch1->nb_orbits == 1 && Sch2->nb_orbits == 1) {
		ost << "The group is transitive on points and on lines.\\\\" << endl;
		std::vector<int> Orb1;
		std::vector<int> Orb2;
		Sch1->get_orbit_in_order(Orb1, 0 /* orbit_idx */, verbose_level);
		Sch2->get_orbit_in_order(Orb2, 0 /* orbit_idx */, verbose_level);

		int *Inc;
		file_io Fio;
		string fname;

		fname.assign(fname_base);
		fname.append("_incma_transitive.csv");

		P->make_incidence_matrix(Orb1, Orb2, Inc, verbose_level);

		Fio.int_matrix_write_csv(fname, Inc, Orb1.size(), Orb2.size());

		FREE_int(Inc);

		int p;
		for (p = 2; p < Orb1.size(); p++) {

			if ((Orb1.size() % p) == 0) {

				cout << "considering subgroup of index " << p << endl;

				int *v, *w;
				std::vector<int> Orb1_subgroup;
				std::vector<int> Orb2_subgroup;
				combinatorics_domain Combi;

				v = NEW_int(Orb1.size());
				w = NEW_int(Orb1.size());
				for (i = 0; i < Orb1.size(); i++) {
					v[i] = Orb1[i];
				}
				Combi.int_vec_splice(v, w, Orb1.size(), p);
				for (i = 0; i < Orb1.size(); i++) {
					Orb1_subgroup.push_back(w[i]);

				}

				for (i = 0; i < Orb1.size(); i++) {
					v[i] = Orb2[i];
				}
				Combi.int_vec_splice(v, w, Orb1.size(), p);
				for (i = 0; i < Orb1.size(); i++) {
					Orb2_subgroup.push_back(w[i]);

				}
				FREE_int(v);
				FREE_int(w);

				fname.assign(fname_base);
				fname.append("_incma_subgroup");
				char str[1000];

				sprintf(str, "_index_%d.csv", p);
				fname.append(str);

				P->make_incidence_matrix(Orb1_subgroup, Orb2_subgroup, Inc, verbose_level);

				Fio.int_matrix_write_csv(fname, Inc, Orb1.size(), Orb2.size());
				FREE_int(Inc);
			}
		}

	}

	ost << "Orbits on points:\\\\" << endl;
	Sch1->print_orbit_lengths_tex(ost);

	ost << "Orbits on lines:\\\\" << endl;
	Sch2->print_orbit_lengths_tex(ost);


	int f_print_subscripts = FALSE;
	ost << "Row scheme:\\\\" << endl;
	Inc->get_and_print_row_tactical_decomposition_scheme_tex(
		ost, TRUE /* f_enter_math */,
		f_print_subscripts, *Stack);
	ost << "Column scheme:\\\\" << endl;
	Inc->get_and_print_column_tactical_decomposition_scheme_tex(
		ost, TRUE /* f_enter_math */,
		f_print_subscripts, *Stack);



	FREE_OBJECT(Sch1);
	FREE_OBJECT(Sch2);
	FREE_OBJECT(Inc);
	FREE_OBJECT(Stack);

	if (f_v) {
		cout << "projective_space_with_action::report_decomposition_by_single_automorphism done" << endl;
	}
}



#if 0
void projective_space_with_action::merge_packings(
		std::string *fnames, int nb_files,
		std::string &file_of_spreads,
		classify_bitvectors *&CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "projective_space_with_action::merge_packings" << endl;
	}

	CB = NEW_OBJECT(classify_bitvectors);


	// for use if INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE
	long int *Spread_table;
	int nb_spreads;
	int spread_size;

	if (f_v) {
		cout << "projective_space_with_action::merge_packings "
				"Reading spread table from file "
				<< file_of_spreads << endl;
	}
	Fio.lint_matrix_read_csv(file_of_spreads,
			Spread_table, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}

	int f, g, N, table_length, nb_reject = 0;

	N = 0;

	if (f_v) {
		cout << "projective_space_with_action::merge_packings "
				"counting the overall number of input packings" << endl;
	}

	for (f = 0; f < nb_files; f++) {

		if (f_v) {
			cout << "projective_space_with_action::merge_packings file "
					<< f << " / " << nb_files << " : " << fnames[f] << endl;
		}

		spreadsheet *S;

		S = NEW_OBJECT(spreadsheet);

		S->read_spreadsheet(fnames[f], 0 /*verbose_level*/);

		table_length = S->nb_rows - 1;
		N += table_length;



		FREE_OBJECT(S);

	}

	if (f_v) {
		cout << "projective_space_with_action::merge_packings file "
				<< "we have " << N << " packings in "
				<< nb_files << " files" << endl;
	}

	for (f = 0; f < nb_files; f++) {

		if (f_v) {
			cout << "projective_space_with_action::merge_packings file "
					<< f << " / " << nb_files << " : " << fnames[f] << endl;
		}

		spreadsheet *S;

		S = NEW_OBJECT(spreadsheet);

		S->read_spreadsheet(fnames[f], 0 /*verbose_level*/);
		if (FALSE /*f_v3*/) {
			S->print_table(cout, FALSE);
			}

		int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
		int nb_rows_idx, nb_cols_idx, canonical_form_idx;

		ago_idx = S->find_by_column("ago");
		original_file_idx = S->find_by_column("original_file");
		input_idx_idx = S->find_by_column("input_idx");
		input_set_idx = S->find_by_column("input_set");
		nb_rows_idx = S->find_by_column("nb_rows");
		nb_cols_idx = S->find_by_column("nb_cols");
		canonical_form_idx = S->find_by_column("canonical_form");

		table_length = S->nb_rows - 1;

		//rep,ago,original_file,input_idx,input_set,nb_rows,nb_cols,canonical_form


		for (g = 0; g < table_length; g++) {

			int ago;
			char *text;
			long int *the_set_in;
			int set_size_in;
			long int *canonical_labeling;
			int canonical_labeling_sz;
			int nb_rows, nb_cols;
			object_in_projective_space *OiP;


			ago = S->get_int(g + 1, ago_idx);
			nb_rows = S->get_int(g + 1, nb_rows_idx);
			nb_cols = S->get_int(g + 1, nb_cols_idx);

			text = S->get_string(g + 1, input_set_idx);
			lint_vec_scan(text, the_set_in, set_size_in);


			if (f_v) {
				cout << "File " << f << " / " << nb_files
						<< ", input set " << g << " / "
						<< table_length << endl;
				//int_vec_print(cout, the_set_in, set_size_in);
				//cout << endl;
				}

			if (FALSE) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (FALSE) {
				cout << "text=" << text << endl;
			}
			lint_vec_scan(text, canonical_labeling, canonical_labeling_sz);
			if (FALSE) {
				cout << "File " << f << " / " << nb_files
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				lint_vec_print(cout, canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "projective_space_with_action::merge_packings "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
				Spread_table, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = TRUE;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "projective_space_with_action::merge_packings "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "projective_space_with_action::merge_packings "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);

			OiP->set_as_string.assign(text);

			int i, j, ii, jj, a;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"before bitvector_allocate_and_coded_length" << endl;
			}
			canonical_form = bitvector_allocate_and_coded_length(
					L, canonical_form_len);
			for (i = 0; i < nb_rows; i++) {
				for (j = 0; j < nb_cols; j++) {
					if (Incma_out[i * nb_cols + j]) {
						a = i * nb_cols + j;
						bitvector_set_bit(canonical_form, a);
						}
					}
				}

			if (CB->n == 0) {
				if (f_v) {
					cout << "projective_space_with_action::merge_packings "
							"before CB->init" << endl;
				}
				CB->init(N, canonical_form_len, verbose_level);
				}
			if (f_v) {
				cout << "projective_space_with_action::merge_packings "
						"before CB->add" << endl;
			}
			int idx;
			int f_found;

			CB->search_and_add_if_new(canonical_form, OiP, f_found, idx, 0 /*verbose_level*/);
			if (f_found) {
				nb_reject++;
			}
			if (f_v) {
				cout << "projective_space_with_action::merge_packings "
						"CB->add returns f_found = " << f_found
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject << endl;
			}


			//int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;


			FREE_lint(the_set_in);
			//FREE_int(canonical_labeling);
			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_uchar(canonical_form);

		} // next g



	} // next f

	if (f_v) {
		cout << "projective_space_with_action::merge_packings done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings" << endl;
		}


	//FREE_OBJECT(CB);
	FREE_lint(Spread_table);

	if (f_v) {
		cout << "projective_space_with_action::merge_packings done" << endl;
	}
}

void projective_space_with_action::select_packings(
		std::string &fname,
		std::string &file_of_spreads_original,
		spread_tables *Spread_tables,
		int f_self_polar,
		int f_ago, int select_ago,
		classify_bitvectors *&CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_accept = 0;
	file_io Fio;

	if (f_v) {
		cout << "projective_space_with_action::select_packings" << endl;
	}

	CB = NEW_OBJECT(classify_bitvectors);



	long int *Spread_table;
	int nb_spreads;
	int spread_size;
	int packing_size;
	int a, b;

	if (f_v) {
		cout << "projective_space_with_action::select_packings "
				"Reading spread table from file "
				<< file_of_spreads_original << endl;
	}
	Fio.lint_matrix_read_csv(file_of_spreads_original,
			Spread_table, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (nb_spreads != Spread_tables->nb_spreads) {
		cout << "projective_space_with_action::select_packings "
				"nb_spreads != Spread_tables->nb_spreads" << endl;
		exit(1);
	}
	if (spread_size != Spread_tables->spread_size) {
		cout << "projective_space_with_action::select_packings "
				"spread_size != Spread_tables->spread_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads_original << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}



	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s" << endl;
	}

	int *s2l, *l2s;
	int i, idx;
	long int *set;
	long int extra_data[1];
	sorting Sorting;

	extra_data[0] = spread_size;

	set = NEW_lint(spread_size);
	s2l = NEW_int(nb_spreads);
	l2s = NEW_int(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		lint_vec_copy(Spread_tables->spread_table +
				i * spread_size, set, spread_size);
		Sorting.lint_vec_heapsort(set, spread_size);
		if (!Sorting.search_general(Spread_tables->spread_table,
				nb_spreads, (void *) set, idx,
				table_of_sets_compare_func,
				extra_data, 0 /*verbose_level*/)) {
			cout << "projective_space_with_action::select_packings "
					"cannot find spread " << i << " = ";
			lint_vec_print(cout, set, spread_size);
			cout << endl;
			exit(1);
		}
		s2l[i] = idx;
		l2s[idx] = i;
	}
	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s done" << endl;
	}

	int g, table_length, nb_reject = 0;


	if (f_v) {
		cout << "projective_space_with_action::select_packings file "
				<< fname << endl;
	}

	spreadsheet *S;

	S = NEW_OBJECT(spreadsheet);

	S->read_spreadsheet(fname, 0 /*verbose_level*/);
	if (FALSE /*f_v3*/) {
		S->print_table(cout, FALSE);
		}

	int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
	int nb_rows_idx, nb_cols_idx, canonical_form_idx;

	ago_idx = S->find_by_column("ago");
	original_file_idx = S->find_by_column("original_file");
	input_idx_idx = S->find_by_column("input_idx");
	input_set_idx = S->find_by_column("input_set");
	nb_rows_idx = S->find_by_column("nb_rows");
	nb_cols_idx = S->find_by_column("nb_cols");
	canonical_form_idx = S->find_by_column("canonical_form");

	table_length = S->nb_rows - 1;

	//rep,ago,original_file,input_idx,
	//input_set,nb_rows,nb_cols,canonical_form
	int f_first = TRUE;


	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		long int *the_set_in;
		int set_size_in;
		long int *canonical_labeling;
		int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP;
		int f_accept = FALSE;
		int *set1;
		int *set2;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		lint_vec_scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;

		if (f_v && (g % 1000) == 0) {
			cout << "File " << fname
					<< ", input set " << g << " / "
					<< table_length << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		if (f_self_polar) {
			set1 = NEW_int(packing_size);
			set2 = NEW_int(packing_size);

			// test if self-polar:
			for (i = 0; i < packing_size; i++) {
				a = the_set_in[i];
				b = s2l[a];
				set1[i] = b;
			}
			Sorting.int_vec_heapsort(set1, packing_size);
			for (i = 0; i < packing_size; i++) {
				a = set1[i];
				b = Spread_tables->dual_spread_idx[a];
				set2[i] = b;
			}
			Sorting.int_vec_heapsort(set2, packing_size);

#if 0
			cout << "set1: ";
			int_vec_print(cout, set1, packing_size);
			cout << endl;
			cout << "set2: ";
			int_vec_print(cout, set2, packing_size);
			cout << endl;
#endif
			if (int_vec_compare(set1, set2, packing_size) == 0) {
				cout << "The packing is self-polar" << endl;
				f_accept = TRUE;
			}
			else {
				f_accept = FALSE;
			}
			FREE_int(set1);
			FREE_int(set2);
		}
		if (f_ago) {
			if (ago == select_ago) {
				f_accept = TRUE;
			}
			else {
				f_accept = FALSE;
			}
		}



		if (f_accept) {

			nb_accept++;


			if (FALSE) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (FALSE) {
				cout << "text=" << text << endl;
			}
			lint_vec_scan(text, canonical_labeling, canonical_labeling_sz);
			if (FALSE) {
				cout << "File " << fname
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				lint_vec_print(cout, canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "projective_space_with_action::select_packings "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (FALSE) {
				cout << "projective_space_with_action::select_packings "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
				Spread_table, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (FALSE) {
				cout << "projective_space_with_action::merge_packings "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = TRUE;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (FALSE) {
				cout << "projective_space_with_action::select_packings "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (FALSE) {
				cout << "projective_space_with_action::select_packings "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "projective_space_with_action::select_packings "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "projective_space_with_action::select_packings "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);

			OiP->set_as_string.assign(text);

			int i, j, ii, jj, a;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (FALSE) {
				cout << "projective_space_with_action::select_packings "
						"before bitvector_allocate_and_coded_length" << endl;
			}
			canonical_form = bitvector_allocate_and_coded_length(
					L, canonical_form_len);
			for (i = 0; i < nb_rows; i++) {
				for (j = 0; j < nb_cols; j++) {
					if (Incma_out[i * nb_cols + j]) {
						a = i * nb_cols + j;
						bitvector_set_bit(canonical_form, a);
						}
					}
				}

			if (f_first) {
				if (f_v) {
					cout << "projective_space_with_action::select_packings "
							"before CB->init" << endl;
				}
				CB->init(table_length, canonical_form_len, verbose_level);
				f_first = FALSE;
			}


			if (f_v) {
				cout << "projective_space_with_action::select_packings "
						"before CB->add" << endl;
			}

			int idx;
			int f_found;

			CB->search_and_add_if_new(canonical_form, OiP, f_found, idx, 0 /*verbose_level*/);
			if (f_found) {
				cout << "reject" << endl;
				nb_reject++;
			}
			if (f_v) {
				cout << "projective_space_with_action::select_packings "
						"CB->add returns f_found = " << f_found
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject
						<< " nb_accept=" << nb_accept
						<< " CB->n=" << CB->n
						<< " CB->nb_types=" << CB->nb_types
						<< endl;
			}


			//int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;

			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_int(canonical_labeling);
			//FREE_uchar(canonical_form);
		} // if (f_accept)



		FREE_lint(the_set_in);

	} // next g




	if (f_v) {
		cout << "projective_space_with_action::select_packings done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings. nb_accept = " << nb_accept
				<< " CB->n = " << CB->n
				<< " CB->nb_types = " << CB->nb_types
				<< endl;
		}


	//FREE_OBJECT(CB);
	FREE_lint(Spread_table);

	if (f_v) {
		cout << "projective_space_with_action::select_packings done" << endl;
	}
}



void projective_space_with_action::select_packings_self_dual(
		std::string &fname,
		std::string &file_of_spreads_original,
		int f_split, int split_r, int split_m,
		spread_tables *Spread_tables,
		classify_bitvectors *&CB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_accept = 0;
	file_io Fio;

	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual" << endl;
	}

	CB = NEW_OBJECT(classify_bitvectors);



	long int *Spread_table_original;
	int nb_spreads;
	int spread_size;
	int packing_size;
	int a, b;

	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"Reading spread table from file "
				<< file_of_spreads_original << endl;
	}
	Fio.lint_matrix_read_csv(file_of_spreads_original,
			Spread_table_original, nb_spreads, spread_size,
			0 /* verbose_level */);
	if (nb_spreads != Spread_tables->nb_spreads) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"nb_spreads != Spread_tables->nb_spreads" << endl;
		exit(1);
	}
	if (spread_size != Spread_tables->spread_size) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"spread_size != Spread_tables->spread_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Reading spread table from file "
				<< file_of_spreads_original << " done" << endl;
		cout << "The spread table contains " << nb_spreads
				<< " spreads" << endl;
	}



	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s" << endl;
	}

	int *s2l, *l2s;
	int i, idx;
	long int *set;
	long int extra_data[1];
	sorting Sorting;

	extra_data[0] = spread_size;

	set = NEW_lint(spread_size);
	s2l = NEW_int(nb_spreads);
	l2s = NEW_int(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		lint_vec_copy(Spread_table_original +
				i * spread_size, set, spread_size);
		Sorting.lint_vec_heapsort(set, spread_size);
		if (!Sorting.search_general(Spread_tables->spread_table,
				nb_spreads, (int *) set, idx,
				table_of_sets_compare_func,
				extra_data, 0 /*verbose_level*/)) {
			cout << "projective_space_with_action::"
					"select_packings_self_dual "
					"cannot find spread " << i << " = ";
			lint_vec_print(cout, set, spread_size);
			cout << endl;
			exit(1);
		}
		s2l[i] = idx;
		l2s[idx] = i;
	}
	if (f_v) {
		cout << "Reading file_isomorphism_type_of_spreads "
				"computing s2l and l2s done" << endl;
	}

	int g, table_length, nb_reject = 0;


	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"file " << fname << endl;
	}

	spreadsheet *S;

	S = NEW_OBJECT(spreadsheet);

	S->read_spreadsheet(fname, 0 /*verbose_level*/);
	if (FALSE /*f_v3*/) {
		S->print_table(cout, FALSE);
		}

	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"read file " << fname << endl;
	}


	int ago_idx, original_file_idx, input_idx_idx, input_set_idx;
	int nb_rows_idx, nb_cols_idx, canonical_form_idx;

	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"finding column indices" << endl;
	}

	ago_idx = S->find_by_column("ago");
	original_file_idx = S->find_by_column("original_file");
	input_idx_idx = S->find_by_column("input_idx");
	input_set_idx = S->find_by_column("input_set");
	nb_rows_idx = S->find_by_column("nb_rows");
	nb_cols_idx = S->find_by_column("nb_cols");
	canonical_form_idx = S->find_by_column("canonical_form");

	table_length = S->nb_rows - 1;

	//rep,ago,original_file,input_idx,
	//input_set,nb_rows,nb_cols,canonical_form
	int f_first = TRUE;


	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"first pass, table_length=" << table_length << endl;
	}

	// first pass: build up the database:

	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		long int *the_set_in;
		int set_size_in;
		long int *canonical_labeling;
		int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP;
		int f_accept;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		lint_vec_scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;

		if (f_v && (g % 1000) == 0) {
			cout << "File " << fname
					<< ", input set " << g << " / "
					<< table_length << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		f_accept = TRUE;



		if (f_accept) {

			nb_accept++;


			if (FALSE) {
				cout << "canonical_form_idx=" << canonical_form_idx << endl;
			}
			text = S->get_string(g + 1, canonical_form_idx);
			if (FALSE) {
				cout << "text=" << text << endl;
			}
			lint_vec_scan(text, canonical_labeling, canonical_labeling_sz);
			if (FALSE) {
				cout << "File " << fname
						<< ", input set " << g << " / "
						<< table_length << " canonical_labeling = ";
				lint_vec_print(cout,
						canonical_labeling, canonical_labeling_sz);
				cout << endl;
				}

			if (canonical_labeling_sz != nb_rows + nb_cols) {
				cout << "projective_space_with_action::select_packings_self_dual "
						"canonical_labeling_sz != nb_rows + nb_cols" << endl;
				exit(1);
			}

			OiP = NEW_OBJECT(object_in_projective_space);

			if (FALSE) {
				cout << "projective_space_with_action::select_packings_self_dual "
						"before init_packing_from_spread_table" << endl;
			}
			OiP->init_packing_from_spread_table(P, the_set_in,
					Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
			if (FALSE) {
				cout << "projective_space_with_action::select_packings_self_dual "
						"after init_packing_from_spread_table" << endl;
			}
			OiP->f_has_known_ago = TRUE;
			OiP->known_ago = ago;

			int *Incma_in;
			int *Incma_out;
			int nb_rows1, nb_cols1;
			int *partition;
			uchar *canonical_form;
			int canonical_form_len;


			if (FALSE) {
				cout << "projective_space_with_action::select_packings_self_dual "
						"before encode_incma" << endl;
			}
			OiP->encode_incma(Incma_in, nb_rows1, nb_cols1,
					partition, 0 /*verbose_level - 1*/);
			if (FALSE) {
				cout << "projective_space_with_action::select_packings_self_dual "
						"after encode_incma" << endl;
			}
			if (nb_rows1 != nb_rows) {
				cout << "projective_space_with_action::select_packings_self_dual "
						"nb_rows1 != nb_rows" << endl;
				exit(1);
			}
			if (nb_cols1 != nb_cols) {
				cout << "projective_space_with_action::select_packings_self_dual "
						"nb_cols1 != nb_cols" << endl;
				exit(1);
			}

			OiP->input_fname = S->get_string(g + 1, original_file_idx);
			OiP->input_idx = S->get_int(g + 1, input_idx_idx);

			text = S->get_string(g + 1, input_set_idx);

			OiP->set_as_string.assign(text);

			int i, j, ii, jj, a;
			int L = nb_rows * nb_cols;

			Incma_out = NEW_int(L);
			for (i = 0; i < nb_rows; i++) {
				ii = canonical_labeling[i];
				for (j = 0; j < nb_cols; j++) {
					jj = canonical_labeling[nb_rows + j] - nb_rows;
					//cout << "i=" << i << " j=" << j << " ii=" << ii
					//<< " jj=" << jj << endl;
					Incma_out[i * nb_cols + j] = Incma_in[ii * nb_cols + jj];
					}
				}
			if (FALSE) {
				cout << "projective_space_with_action::select_packings_self_dual "
						"before bitvector_allocate_and_coded_length" << endl;
			}
			canonical_form = bitvector_allocate_and_coded_length(
					L, canonical_form_len);
			for (i = 0; i < nb_rows; i++) {
				for (j = 0; j < nb_cols; j++) {
					if (Incma_out[i * nb_cols + j]) {
						a = i * nb_cols + j;
						bitvector_set_bit(canonical_form, a);
						}
					}
				}

			if (f_first) {
				if (f_v) {
					cout << "projective_space_with_action::select_packings_self_dual "
							"before CB->init" << endl;
				}
				CB->init(table_length, canonical_form_len, verbose_level);
				f_first = FALSE;
			}


			if (FALSE) {
				cout << "projective_space_with_action::select_packings_self_dual "
						"before CB->add" << endl;
			}

			int idx;
			int f_found;

			CB->search_and_add_if_new(canonical_form, OiP, f_found, idx, 0 /*verbose_level*/);
			if (f_found) {
				cout << "reject" << endl;
				nb_reject++;
			}
			if (FALSE) {
				cout << "projective_space_with_action::select_packings_self_dual "
						"CB->add f_found = " << f_found
						<< " nb iso = " << CB->nb_types
						<< " nb_reject=" << nb_reject
						<< " nb_accept=" << nb_accept
						<< " CB->n=" << CB->n
						<< " CB->nb_types=" << CB->nb_types
						<< endl;
			}


			//int idx;

			object_in_projective_space_with_action *OiPA;

			OiPA = NEW_OBJECT(object_in_projective_space_with_action);

			OiPA->init(OiP, ago, nb_rows, nb_cols,
					canonical_labeling, 0 /*verbose_level*/);
			idx = CB->type_of[CB->n - 1];
			CB->Type_extra_data[idx] = OiPA;

			FREE_int(Incma_in);
			FREE_int(Incma_out);
			FREE_int(partition);
			//FREE_int(canonical_labeling);
			//FREE_uchar(canonical_form);
		} // if (f_accept)



		FREE_lint(the_set_in);

	} // next g




	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual done, "
				"we found " << CB->nb_types << " isomorphism types "
				"of packings. nb_accept = " << nb_accept
				<< " CB->n = " << CB->n
				<< " CB->nb_types = " << CB->nb_types
				<< endl;
		}


	// second pass:

	int nb_self_dual = 0;
	int g1 = 0;
	int *self_dual_cases;
	int nb_self_dual_cases = 0;


	self_dual_cases = NEW_int(table_length);


	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"second pass, table_length="
				<< table_length << endl;
	}


	for (g = 0; g < table_length; g++) {

		int ago;
		char *text;
		int *the_set_in;
		int set_size_in;
		int *canonical_labeling1;
		int *canonical_labeling2;
		//int canonical_labeling_sz;
		int nb_rows, nb_cols;
		object_in_projective_space *OiP1;
		object_in_projective_space *OiP2;
		long int *set1;
		long int *set2;

		ago = S->get_int(g + 1, ago_idx);
		nb_rows = S->get_int(g + 1, nb_rows_idx);
		nb_cols = S->get_int(g + 1, nb_cols_idx);

		text = S->get_string(g + 1, input_set_idx);
		int_vec_scan(text, the_set_in, set_size_in);

		packing_size = set_size_in;


		if (f_split) {
			if ((g % split_m) != split_r) {
				continue;
			}
		}
		g1++;
		if (f_v && (g1 % 100) == 0) {
			cout << "File " << fname
					<< ", case " << g1 << " input set " << g << " / "
					<< table_length
					<< " nb_self_dual=" << nb_self_dual << endl;
			//int_vec_print(cout, the_set_in, set_size_in);
			//cout << endl;
			}


		set1 = NEW_lint(packing_size);
		set2 = NEW_lint(packing_size);

		for (i = 0; i < packing_size; i++) {
			a = the_set_in[i];
			b = s2l[a];
			set1[i] = b;
		}
		Sorting.lint_vec_heapsort(set1, packing_size);
		for (i = 0; i < packing_size; i++) {
			a = set1[i];
			b = Spread_tables->dual_spread_idx[a];
			set2[i] = l2s[b];
		}
		for (i = 0; i < packing_size; i++) {
			a = set1[i];
			b = l2s[a];
			set1[i] = b;
		}
		Sorting.lint_vec_heapsort(set1, packing_size);
		Sorting.lint_vec_heapsort(set2, packing_size);

#if 0
		cout << "set1: ";
		int_vec_print(cout, set1, packing_size);
		cout << endl;
		cout << "set2: ";
		int_vec_print(cout, set2, packing_size);
		cout << endl;
#endif




		OiP1 = NEW_OBJECT(object_in_projective_space);
		OiP2 = NEW_OBJECT(object_in_projective_space);

		if (FALSE) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"before init_packing_from_spread_table" << endl;
		}
		OiP1->init_packing_from_spread_table(P, set1,
				Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
		OiP2->init_packing_from_spread_table(P, set2,
				Spread_table_original, nb_spreads, spread_size,
				0 /*verbose_level*/);
		if (FALSE) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"after init_packing_from_spread_table" << endl;
		}
		OiP1->f_has_known_ago = TRUE;
		OiP1->known_ago = ago;



		uchar *canonical_form1;
		uchar *canonical_form2;
		int canonical_form_len;



		int *Incma_in1;
		int *Incma_out1;
		int *Incma_in2;
		int *Incma_out2;
		int nb_rows1, nb_cols1;
		int *partition;
		//uchar *canonical_form1;
		//uchar *canonical_form2;
		//int canonical_form_len;


		if (FALSE) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"before encode_incma" << endl;
		}
		OiP1->encode_incma(Incma_in1, nb_rows1, nb_cols1,
				partition, 0 /*verbose_level - 1*/);
		OiP2->encode_incma(Incma_in2, nb_rows1, nb_cols1,
				partition, 0 /*verbose_level - 1*/);
		if (FALSE) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"after encode_incma" << endl;
		}
		if (nb_rows1 != nb_rows) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"nb_rows1 != nb_rows" << endl;
			exit(1);
		}
		if (nb_cols1 != nb_cols) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"nb_cols1 != nb_cols" << endl;
			exit(1);
		}


		if (FALSE) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"before PA->set_stabilizer_of_object" << endl;
			}


		canonical_labeling1 = NEW_int(nb_rows * nb_cols);
		canonical_labeling2 = NEW_int(nb_rows * nb_cols);

		canonical_labeling(
				OiP1,
				canonical_labeling1,
				0 /*verbose_level - 2*/);
		canonical_labeling(
				OiP2,
				canonical_labeling2,
				0 /*verbose_level - 2*/);


		OiP1->input_fname = S->get_string(g + 1, original_file_idx);
		OiP1->input_idx = S->get_int(g + 1, input_idx_idx);
		OiP2->input_fname = S->get_string(g + 1, original_file_idx);
		OiP2->input_idx = S->get_int(g + 1, input_idx_idx);

		text = S->get_string(g + 1, input_set_idx);

		OiP1->set_as_string.assign(text);

		OiP2->set_as_string.assign(text);

		int i, j, ii, jj, a, ret;
		int L = nb_rows * nb_cols;

		Incma_out1 = NEW_int(L);
		Incma_out2 = NEW_int(L);
		for (i = 0; i < nb_rows; i++) {
			ii = canonical_labeling1[i];
			for (j = 0; j < nb_cols; j++) {
				jj = canonical_labeling1[nb_rows + j] - nb_rows;
				//cout << "i=" << i << " j=" << j << " ii=" << ii
				//<< " jj=" << jj << endl;
				Incma_out1[i * nb_cols + j] = Incma_in1[ii * nb_cols + jj];
				}
			}
		for (i = 0; i < nb_rows; i++) {
			ii = canonical_labeling2[i];
			for (j = 0; j < nb_cols; j++) {
				jj = canonical_labeling2[nb_rows + j] - nb_rows;
				//cout << "i=" << i << " j=" << j << " ii=" << ii
				//<< " jj=" << jj << endl;
				Incma_out2[i * nb_cols + j] = Incma_in2[ii * nb_cols + jj];
				}
			}
		if (FALSE) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"before bitvector_allocate_and_coded_length" << endl;
		}
		canonical_form1 = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out1[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form1, a);
					}
				}
			}
		canonical_form2 = bitvector_allocate_and_coded_length(
				L, canonical_form_len);
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				if (Incma_out2[i * nb_cols + j]) {
					a = i * nb_cols + j;
					bitvector_set_bit(canonical_form2, a);
					}
				}
			}


		if (FALSE) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"before CB->search" << endl;
		}

		int idx1, idx2;

		ret = CB->search(canonical_form1, idx1, 0 /*verbose_level*/);

		if (ret == FALSE) {
			cout << "cannot find the dual packing, "
					"something is wrong" << endl;
			ret = CB->search(canonical_form1, idx1, 5 /* verbose_level*/);
#if 0
			cout << "CB:" << endl;
			CB->print_table();
			cout << "canonical form1: ";
			for (int j = 0; j < canonical_form_len; j++) {
				cout << (int) canonical_form1[j];
				if (j < canonical_form_len - 1) {
					cout << ", ";
					}
				}
			cout << endl;
#endif
			exit(1);
		}
		if (FALSE) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"CB->search returns idx1=" << idx1 << endl;
		}
		ret = CB->search(canonical_form2, idx2, 0 /*verbose_level*/);

		if (ret == FALSE) {
			cout << "cannot find the dual packing, "
					"something is wrong" << endl;
			ret = CB->search(canonical_form2, idx2, 5 /* verbose_level*/);
#if 0
			cout << "CB:" << endl;
			CB->print_table();
			cout << "canonical form2: ";
			for (int j = 0; j < canonical_form_len; j++) {
				cout << (int) canonical_form2[j];
				if (j < canonical_form_len - 1) {
					cout << ", ";
					}
				}
#endif
			exit(1);
		}
		if (FALSE) {
			cout << "projective_space_with_action::select_packings_self_dual "
					"CB->search returns idx2=" << idx2 << endl;
		}

		FREE_int(Incma_in1);
		FREE_int(Incma_out1);
		FREE_int(Incma_in2);
		FREE_int(Incma_out2);
		FREE_int(partition);
		FREE_int(canonical_labeling1);
		FREE_int(canonical_labeling2);
		FREE_uchar(canonical_form1);
		FREE_uchar(canonical_form2);

		FREE_lint(set1);
		FREE_lint(set2);

		if (idx1 == idx2) {
			cout << "self-dual" << endl;
			nb_self_dual++;
			self_dual_cases[nb_self_dual_cases++] = g;
		}

		FREE_int(the_set_in);

	} // next g

	string fname_base;
	string fname_self_dual;
	char str[1000];

	fname_base.assign(fname);
	chop_off_extension(fname_base);
	fname_self_dual.assign(fname);
	chop_off_extension(fname_self_dual);
	if (f_split) {
		sprintf(str, "_self_dual_r%d_m%d.csv", split_r, split_m);
	}
	else {
		sprintf(str, "_self_dual.csv");
	}
	fname_self_dual.append(str);
	cout << "saving self_dual_cases to file " << fname_self_dual << endl;
	Fio.int_vec_write_csv(self_dual_cases, nb_self_dual_cases,
			fname_self_dual, "self_dual_idx");
	cout << "written file " << fname_self_dual
			<< " of size " << Fio.file_size(fname_self_dual) << endl;



	//FREE_OBJECT(CB);
	FREE_lint(Spread_table_original);

	if (f_v) {
		cout << "projective_space_with_action::select_packings_self_dual "
				"done, nb_self_dual = " << nb_self_dual << endl;
	}
}
#endif


object_in_projective_space *
projective_space_with_action::create_object_from_string(
	int type, std::string &input_fname, int input_idx,
	std::string &set_as_string, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::create_object_from_string" << endl;
		cout << "type=" << type << endl;
	}


	object_in_projective_space *OiP;

	OiP = NEW_OBJECT(object_in_projective_space);

	OiP->init_object_from_string(P,
			type, input_fname, input_idx,
			set_as_string, verbose_level);


	if (f_v) {
		cout << "projective_space_with_action::create_object_from_string"
				" done" << endl;
	}
	return OiP;
}

object_in_projective_space *
projective_space_with_action::create_object_from_int_vec(
	int type, std::string &input_fname, int input_idx,
	long int *the_set, int set_sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::create_object_from_int_vec" << endl;
		cout << "type=" << type << endl;
	}


	object_in_projective_space *OiP;

	OiP = NEW_OBJECT(object_in_projective_space);

	OiP->init_object_from_int_vec(P,
			type, input_fname, input_idx,
			the_set, set_sz, verbose_level);


	if (f_v) {
		cout << "projective_space_with_action::create_object_from_int_vec"
				" done" << endl;
	}
	return OiP;
}


void projective_space_with_action::compute_group_of_set(long int *set, int set_sz,
		strong_generators *&Sg,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	long int a;
	char str[1000];

	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set" << endl;
	}

	projective_space_object_classifier_description *Descr;
	projective_space_object_classifier *Classifier;

	Descr = NEW_OBJECT(projective_space_object_classifier_description);
	Classifier = NEW_OBJECT(projective_space_object_classifier);

	Descr->f_input = TRUE;
	Descr->Data = NEW_OBJECT(data_input_stream);
	Descr->Data->input_type[Descr->Data->nb_inputs] = INPUT_TYPE_SET_OF_POINTS;
	Descr->Data->input_string[Descr->Data->nb_inputs].assign("");
	for (i = 0; i < set_sz; i++) {
		a = set[i];
		sprintf(str, "%ld", a);
		Descr->Data->input_string[Descr->Data->nb_inputs].append(str);
		if (i < set_sz - 1) {
			Descr->Data->input_string[Descr->Data->nb_inputs].append(",");
		}
	}
	Descr->Data->input_string2[Descr->Data->nb_inputs].assign("");
	Descr->Data->nb_inputs++;

	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set before Classifier->do_the_work" << endl;
	}

	Classifier->do_the_work(
			Descr,
			this,
			verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set after Classifier->do_the_work" << endl;
	}

	int idx;
	long int ago;

	idx = Classifier->CB->type_of[Classifier->CB->n - 1];


	object_in_projective_space_with_action *OiPA;

	OiPA = (object_in_projective_space_with_action *) Classifier->CB->Type_extra_data[idx];


	ago = OiPA->ago;

	Sg = OiPA->Aut_gens;

	Sg->A = A;


	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set ago = " << ago << endl;
	}



	if (f_v) {
		cout << "projective_space_with_action::compute_group_of_set done" << endl;
	}
}


void projective_space_with_action::map(formula *Formula,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::map" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::map" << endl;
		cout << "formula:" << endl;
		Formula->print();
	}

	if (!Formula->f_is_homogeneous) {
		cout << "Formula is not homogeneous" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Formula is homogeneous of degree " << Formula->degree << endl;
		exit(1);
	}
	if (Formula->nb_managed_vars != P->n + 1) {
		cout << "Formula->nb_managed_vars != P->n + 1" << endl;
		exit(1);
	}

	homogeneous_polynomial_domain *Poly;

	Poly = NEW_OBJECT(homogeneous_polynomial_domain);

	if (f_v) {
		cout << "projective_space_with_action::map before Poly->init" << endl;
	}
	Poly->init(F,
			Formula->nb_managed_vars /* nb_vars */, Formula->degree,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::map after Poly->init" << endl;
	}


	syntax_tree_node **Subtrees;
	int nb_monomials;

	if (f_v) {
		cout << "projective_space_with_action::map before Formula->get_subtrees" << endl;
	}
	Formula->get_subtrees(Poly, Subtrees, nb_monomials, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::map after Formula->get_subtrees" << endl;
	}

	int i;

	for (i = 0; i < nb_monomials; i++) {
		cout << "Monomial " << i << " : ";
		if (Subtrees[i]) {
			Subtrees[i]->print_expression(cout);
			cout << " * ";
			Poly->print_monomial(cout, i);
			cout << endl;
		}
		else {
			cout << "no subtree" << endl;
		}
	}


	int *Coefficient_vector;

	Coefficient_vector = NEW_int(nb_monomials);

	Formula->evaluate(Poly,
			Subtrees, evaluate_text, Coefficient_vector,
			verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::map coefficient vector:" << endl;
		Orbiter->Int_vec.print(cout, Coefficient_vector, nb_monomials);
		cout << endl;
	}

#if 0
	del_pezzo_surface_of_degree_two_domain *del_Pezzo;

	del_Pezzo = NEW_OBJECT(del_pezzo_surface_of_degree_two_domain);

	del_Pezzo->init(P, Poly4_3, verbose_level);

	del_pezzo_surface_of_degree_two_object *del_Pezzo_surface;

	del_Pezzo_surface = NEW_OBJECT(del_pezzo_surface_of_degree_two_object);

	del_Pezzo_surface->init(del_Pezzo,
			Formula, Subtrees, Coefficient_vector,
			verbose_level);

	del_Pezzo_surface->enumerate_points_and_lines(verbose_level);

	del_Pezzo_surface->pal->write_points_to_txt_file(Formula->name_of_formula, verbose_level);

	del_Pezzo_surface->create_latex_report(Formula->name_of_formula, Formula->name_of_formula_latex, verbose_level);

	FREE_OBJECT(del_Pezzo_surface);
	FREE_OBJECT(del_Pezzo);
#endif

	FREE_int(Coefficient_vector);
	FREE_OBJECT(Poly);

	if (f_v) {
		cout << "projective_space_with_action::map done" << endl;
	}
}


void projective_space_with_action::analyze_del_Pezzo_surface(formula *Formula,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface" << endl;
	}

	if (f_v) {
		cout << "projective_space_activity::analyze_del_Pezzo_surface" << endl;
		cout << "formula:" << endl;
		Formula->print();
	}

	if (!Formula->f_is_homogeneous) {
		cout << "Formula is not homogeneous" << endl;
		exit(1);
	}
	if (Formula->degree != 4) {
		cout << "Formula is not of degree 4. Degree is " << Formula->degree << endl;
		exit(1);
	}
	if (Formula->nb_managed_vars != 3) {
		cout << "Formula should have 3 managed variables. Has " << Formula->nb_managed_vars << endl;
		exit(1);
	}

	homogeneous_polynomial_domain *Poly4_3;

	Poly4_3 = NEW_OBJECT(homogeneous_polynomial_domain);

	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface before Poly->init" << endl;
	}
	Poly4_3->init(F,
			Formula->nb_managed_vars /* nb_vars */, Formula->degree,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface after Poly->init" << endl;
	}


	syntax_tree_node **Subtrees;
	int nb_monomials;

	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface before Formula->get_subtrees" << endl;
	}
	Formula->get_subtrees(Poly4_3, Subtrees, nb_monomials, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface after Formula->get_subtrees" << endl;
	}

	int i;

	for (i = 0; i < nb_monomials; i++) {
		cout << "Monomial " << i << " : ";
		if (Subtrees[i]) {
			Subtrees[i]->print_expression(cout);
			cout << " * ";
			Poly4_3->print_monomial(cout, i);
			cout << endl;
		}
		else {
			cout << "no subtree" << endl;
		}
	}


	int *Coefficient_vector;

	Coefficient_vector = NEW_int(nb_monomials);

	Formula->evaluate(Poly4_3,
			Subtrees, evaluate_text, Coefficient_vector,
			verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface coefficient vector:" << endl;
		Orbiter->Int_vec.print(cout, Coefficient_vector, nb_monomials);
		cout << endl;
	}

	del_pezzo_surface_of_degree_two_domain *del_Pezzo;

	del_Pezzo = NEW_OBJECT(del_pezzo_surface_of_degree_two_domain);

	del_Pezzo->init(P, Poly4_3, verbose_level);

	del_pezzo_surface_of_degree_two_object *del_Pezzo_surface;

	del_Pezzo_surface = NEW_OBJECT(del_pezzo_surface_of_degree_two_object);

	del_Pezzo_surface->init(del_Pezzo,
			Formula, Subtrees, Coefficient_vector,
			verbose_level);

	del_Pezzo_surface->enumerate_points_and_lines(verbose_level);

	del_Pezzo_surface->pal->write_points_to_txt_file(Formula->name_of_formula, verbose_level);

	del_Pezzo_surface->create_latex_report(Formula->name_of_formula, Formula->name_of_formula_latex, verbose_level);

	FREE_OBJECT(del_Pezzo_surface);
	FREE_OBJECT(del_Pezzo);

	FREE_int(Coefficient_vector);
	FREE_OBJECT(Poly4_3);

	if (f_v) {
		cout << "projective_space_with_action::analyze_del_Pezzo_surface done" << endl;
	}
}


void projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG(
		int decomposition_by_element_power,
		std::string &decomposition_by_element_data, std::string &fname_base,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG verbose_level="
				<< verbose_level << endl;
	}


	finite_field *F;

	F = P->F;


	{
		char title[1000];
		char author[1000];

		snprintf(title, 1000, "Cheat Sheet PG($%d,%d$)", n, F->q);
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname_base);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG f_decomposition_by_element" << endl;
			}

			int *Elt;

			Elt = NEW_int(A->elt_size_in_int);


			A->make_element_from_string(Elt,
					decomposition_by_element_data, verbose_level);


			A->element_power_int_in_place(Elt,
					decomposition_by_element_power, verbose_level);

			report_decomposition_by_single_automorphism(
					Elt, ost, fname_base,
					verbose_level);

			FREE_int(Elt);


			L.foot(ost);

		}
		file_io Fio;

		if (f_v) {
			cout << "written file " << fname_base << " of size "
					<< Fio.file_size(fname_base) << endl;
		}
	}

	if (f_v) {
		cout << "projective_space_with_action::do_cheat_sheet_for_decomposition_by_element_PG done" << endl;
	}

}


void projective_space_with_action::report(
	ostream &ost,
	layered_graph_draw_options *O,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::report" << endl;
	}



	if (f_v) {
		cout << "projective_space_with_action::report done" << endl;
	}
}


void projective_space_with_action::create_quartic_curve(
		quartic_curve_create_description *Quartic_curve_descr,
		quartic_curve_create *&QC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve" << endl;
	}
	QC = NEW_OBJECT(quartic_curve_create);

	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve before SC->init" << endl;
	}
	QC->init(Quartic_curve_descr, this, QCDA, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve after SC->init" << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve "
				"before SC->apply_transformations" << endl;
	}
	QC->apply_transformations(Quartic_curve_descr->transform_coeffs,
			Quartic_curve_descr->f_inverse_transform,
			verbose_level - 2);

	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve "
				"after SC->apply_transformations" << endl;
	}

	QC->F->PG_element_normalize(QC->QO->eqn15, 1, 15);

	if (f_v) {
		cout << "projective_space_with_action::create_quartic_curve done" << endl;
	}
}

void projective_space_with_action::canonical_form_of_code(
		std::string &label, int m, int n,
		std::string &data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code" << endl;
	}
	int *genma;
	int sz;
	int i, j;
	int *v;
	long int *set;

	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code before int_vec_scan" << endl;
	}
	Orbiter->Int_vec.scan(data, genma, sz);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code after int_vec_scan, sz=" << sz << endl;
	}

	if (sz != m * n) {
		cout << "projective_space_with_action::canonical_form_of_code sz != m * n" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "genma: " << endl;
		Orbiter->Int_vec.print(cout, genma, sz);
		cout << endl;
	}
	v = NEW_int(m);
	set = NEW_lint(n);
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			v[i] = genma[i * n + j];
		}
		if (f_v) {
			cout << "projective_space_with_action::canonical_form_of_code before PA->P->rank_point" << endl;
			Orbiter->Int_vec.print(cout, v, m);
			cout << endl;
		}
		if (P == NULL) {
			cout << "P == NULL" << endl;
			exit(1);
		}
		set[j] = P->rank_point(v);
	}
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code set=";
		Orbiter->Lint_vec.print(cout, set, n);
		cout << endl;
	}

	projective_space_object_classifier_description Descr;
	data_input_stream Data;
	string points_as_string;
	char str[1000];

	sprintf(str, "%ld", set[0]);
	points_as_string.assign(str);
	for (i = 1; i < n; i++) {
		points_as_string.append(",");
		sprintf(str, "%ld", set[i]);
		points_as_string.append(str);
	}
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code points_as_string=" << points_as_string << endl;
	}

	Descr.f_input = TRUE;
	Descr.Data = &Data;

	Descr.f_save_classification = TRUE;
	Descr.save_prefix.assign("code_");

	Descr.f_report = TRUE;
	Descr.report_prefix.assign("code_");
	Descr.report_prefix.append(label);

	Descr.f_classification_prefix = TRUE;
	Descr.classification_prefix.assign("classify_code_");
	Descr.classification_prefix.append(label);

	Data.nb_inputs = 0;
	Data.input_type[Data.nb_inputs] = INPUT_TYPE_SET_OF_POINTS;
	Data.input_string[Data.nb_inputs] = points_as_string;
	Data.nb_inputs++;


	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code before PA->canonical_form" << endl;
	}

	canonical_form(&Descr, verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code after PA->canonical_form" << endl;
	}


	FREE_int(v);
	FREE_lint(set);
	if (f_v) {
		cout << "projective_space_with_action::canonical_form_of_code done" << endl;
	}

}

void projective_space_with_action::table_of_quartic_curves(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::table_of_quartic_curves" << endl;
	}

	if (n != 2) {
		cout << "projective_space_with_action::table_of_quartic_curves we need a two-dimensional projective space" << endl;
		exit(1);
	}

	knowledge_base K;

	int nb_quartic_curves;
	int h;
	quartic_curve_create **QC;
	int *nb_K;
	long int *Table;
	int nb_cols = 6;

	nb_quartic_curves = K.quartic_curves_nb_reps(q);

	QC = (quartic_curve_create **) NEW_pvoid(nb_quartic_curves);

	nb_K = NEW_int(nb_quartic_curves);

	Table = NEW_lint(nb_quartic_curves * nb_cols);

	for (h = 0; h < nb_quartic_curves; h++) {

		if (f_v) {
			cout << "projective_space_with_action::table_of_quartic_curves " << h << " / " << nb_quartic_curves << endl;
		}
		quartic_curve_create_description Quartic_curve_descr;

		Quartic_curve_descr.f_q = TRUE;
		Quartic_curve_descr.q = q;
		Quartic_curve_descr.f_catalogue = TRUE;
		Quartic_curve_descr.iso = h;



		create_quartic_curve(
					&Quartic_curve_descr,
					QC[h],
					verbose_level);

		nb_K[h] = QC[h]->QO->QP->nb_Kowalevski;


		Table[h * nb_cols + 0] = h;
		Table[h * nb_cols + 1] = nb_K[h];
		Table[h * nb_cols + 2] = QC[h]->QO->QP->nb_Kowalevski_on;
		Table[h * nb_cols + 3] = QC[h]->QO->QP->nb_Kowalevski_off;
		Table[h * nb_cols + 4] = QC[h]->QOA->Aut_gens->group_order_as_lint();
		Table[h * nb_cols + 5] = QC[h]->QO->nb_pts;

	}

	file_io Fio;
	char str[1000];

	sprintf(str, "_q%d", q);

	string fname;
	fname.assign("quartic_curves");
	fname.append(str);
	fname.append("_info.csv");

	//Fio.lint_matrix_write_csv(fname, Table, nb_quartic_curves, nb_cols);

	{
		ofstream f(fname);
		int i, j;

		f << "Row,OCN,K,Kon,Koff,Ago,NbPts,BisecantType,Eqn15,Eqn,Pts,Bitangents28";
		f << endl;
		for (i = 0; i < nb_quartic_curves; i++) {
			f << i;
			for (j = 0; j < nb_cols; j++) {
				f << "," << Table[i * nb_cols + j];
			}
			{
				string str;
				f << ",";
				Orbiter->Int_vec.create_string_with_quotes(str, QC[i]->QO->QP->line_type_distribution, 3);
				f << str;
			}
			{
				string str;
				f << ",";
				Orbiter->Int_vec.create_string_with_quotes(str, QC[i]->QO->eqn15, 15);
				f << str;
			}

			{
				stringstream sstr;
				string str;
				QC[i]->QCDA->Dom->print_equation_maple(sstr, QC[i]->QO->eqn15);
				str.assign(sstr.str());
				f << ",";
				f << "\"$";
				f << str;
				f << "$\"";
			}

			{
				string str;
				f << ",";
				Orbiter->Lint_vec.create_string_with_quotes(str, QC[i]->QO->Pts, QC[i]->QO->nb_pts);
				f << str;
			}
			{
				string str;
				f << ",";
				Orbiter->Lint_vec.create_string_with_quotes(str, QC[i]->QO->bitangents28, 28);
				f << str;
			}
			f << endl;
		}
		f << "END" << endl;
	}


	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "projective_space_with_action::table_of_quartic_curves done" << endl;
	}

}

void projective_space_with_action::table_of_cubic_surfaces(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::table_of_cubic_surfaces" << endl;
	}

	if (n != 3) {
		cout << "projective_space_with_action::table_of_cubic_surfaces we need a three-dimensional projective space" << endl;
		exit(1);
	}

	surface_with_action *Surf_A;

	setup_surface_with_action(
			Surf_A,
			verbose_level);


	knowledge_base K;

	int nb_cubic_surfaces;
	int h;
	surface_create **SC;
	int *nb_E;
	long int *Table;
	int nb_cols = 5;


	poset_classification_control Control_six_arcs;


	nb_cubic_surfaces = K.cubic_surface_nb_reps(q);

	SC = (surface_create **) NEW_pvoid(nb_cubic_surfaces);

	nb_E = NEW_int(nb_cubic_surfaces);

	Table = NEW_lint(nb_cubic_surfaces * nb_cols);



	for (h = 0; h < nb_cubic_surfaces; h++) {

		if (f_v) {
			cout << "projective_space_with_action::table_of_cubic_surfaces " << h << " / " << nb_cubic_surfaces << endl;
		}
		surface_create_description Surface_create_description;

		Surface_create_description.f_q = TRUE;
		Surface_create_description.q = q;
		Surface_create_description.f_catalogue = TRUE;
		Surface_create_description.iso = h;


		if (f_v) {
			cout << "projective_space_with_action::table_of_cubic_surfaces before create_surface" << endl;
		}
		Surf_A->create_surface(
				&Surface_create_description,
				SC[h],
				verbose_level);
		if (f_v) {
			cout << "projective_space_with_action::table_of_cubic_surfaces after create_surface" << endl;
		}



		nb_E[h] = SC[h]->SO->SOP->nb_Eckardt_points;


		Table[h * nb_cols + 0] = h;
		Table[h * nb_cols + 1] = nb_E[h];
		if (SC[h]->f_has_group) {
			Table[h * nb_cols + 2] = SC[h]->Sg->group_order_as_lint();
		}
		else {
			Table[h * nb_cols + 2] = 0;
		}
		Table[h * nb_cols + 3] = SC[h]->SO->nb_pts;
		Table[h * nb_cols + 4] = SC[h]->SO->nb_lines;

	}

	file_io Fio;
	char str[1000];

	sprintf(str, "_q%d", q);

	string fname;
	fname.assign("table_of_cubic_surfaces");
	fname.append(str);
	fname.append("_info.csv");

	//Fio.lint_matrix_write_csv(fname, Table, nb_quartic_curves, nb_cols);

	{
		ofstream f(fname);
		int i, j;

		f << "Row,OCN,nbE,Ago,NbPts,NbLines,Eqn20,Eqn,Lines";
		f << endl;
		for (i = 0; i < nb_cubic_surfaces; i++) {
			f << i;
			for (j = 0; j < nb_cols; j++) {
				f << "," << Table[i * nb_cols + j];
			}
			{
				string str;
				f << ",";
				Orbiter->Int_vec.create_string_with_quotes(str, SC[i]->SO->eqn, 20);
				f << str;
			}

			{
				stringstream sstr;
				string str;
				SC[i]->Surf->print_equation_maple(sstr, SC[i]->SO->eqn);
				str.assign(sstr.str());
				f << ",";
				f << "\"$";
				f << str;
				f << "$\"";
			}
			{
				string str;
				f << ",";
				Orbiter->Lint_vec.create_string_with_quotes(str, SC[i]->SO->Lines, SC[i]->SO->nb_lines);
				f << str;
			}

#if 0
			{
				string str;
				f << ",";
				Orbiter->Int_vec.create_string_with_quotes(str, SC[i]->SO->eqn15, 15);
				f << str;
			}


			{
				string str;
				f << ",";
				Orbiter->Lint_vec.create_string_with_quotes(str, SC[i]->SO->Pts, SC[i]->SO->nb_pts);
				f << str;
			}
			{
				string str;
				f << ",";
				Orbiter->Lint_vec.create_string_with_quotes(str, SC[i]->SO->bitangents28, 28);
				f << str;
			}
#endif
			f << endl;
		}
		f << "END" << endl;
	}


	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "projective_space_with_action::table_of_cubic_surfaces done" << endl;
	}

}

void projective_space_with_action::conic_type(
		long int *Pts, int nb_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::conic_type" << endl;
	}


	long int **Pts_on_conic;
	int **Conic_eqn;
	int *nb_pts_on_conic;
	int len;
	int h;


	if (f_v) {
		cout << "projective_space_with_action::conic_type before PA->P->conic_type" << endl;
	}

	P->conic_type(Pts, nb_pts,
			Pts_on_conic, Conic_eqn, nb_pts_on_conic, len,
			verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::conic_type after PA->P->conic_type" << endl;
	}


	cout << "We found the following conics:" << endl;
	for (h = 0; h < len; h++) {
		cout << h << " : " << nb_pts_on_conic[h] << " : ";
		Orbiter->Int_vec.print(cout, Conic_eqn[h], 6);
		cout << " : ";
		Orbiter->Lint_vec.print(cout, Pts_on_conic[h], nb_pts_on_conic[h]);
		cout << endl;
	}

	cout << "computing intersection types with bisecants of the first 11 points:" << endl;
	int Line_P1[55];
	int Line_P2[55];
	int P1, P2;
	long int p1, p2, line_rk;
	long int *pts_on_line;
	long int pt;
	int *Conic_line_intersection_sz;
	int cnt;
	int i, j, q, u, v;
	int nb_pts_per_line;

	q = P->F->q;
	nb_pts_per_line = q + 1;
	pts_on_line = NEW_lint(55 * nb_pts_per_line);

	cnt = 0;
	for (i = 0; i < 11; i++) {
		for (j = i + 1; j < 11; j++) {
			Line_P1[cnt] = i;
			Line_P2[cnt] = j;
			cnt++;
		}
	}
	if (cnt != 55) {
		cout << "cnt != 55" << endl;
		cout << "cnt = " << cnt << endl;
		exit(1);
	}
	for (u = 0; u < 55; u++) {
		P1 = Line_P1[u];
		P2 = Line_P2[u];
		p1 = Pts[P1];
		p2 = Pts[P2];
		line_rk = P->line_through_two_points(p1, p2);
		P->create_points_on_line(line_rk, pts_on_line + u * nb_pts_per_line, 0 /*verbose_level*/);
	}

	Conic_line_intersection_sz = NEW_int(len * 55);
	Orbiter->Int_vec.zero(Conic_line_intersection_sz, len * 55);

	for (h = 0; h < len; h++) {
		for (u = 0; u < 55; u++) {
			for (v = 0; v < nb_pts_per_line; v++) {
				if (P->test_if_conic_contains_point(Conic_eqn[h], pts_on_line[u * nb_pts_per_line + v])) {
					Conic_line_intersection_sz[h * 55 + u]++;
				}

			}
		}
	}

	sorting Sorting;
	int idx;

	cout << "We found the following conics and their intersections with the 55 bisecants:" << endl;
	for (h = 0; h < len; h++) {
		cout << h << " : " << nb_pts_on_conic[h] << " : ";
		Orbiter->Int_vec.print(cout, Conic_eqn[h], 6);
		cout << " : ";
		Orbiter->Int_vec.print_fully(cout, Conic_line_intersection_sz + h * 55, 55);
		cout << " : ";
		Orbiter->Lint_vec.print(cout, Pts_on_conic[h], nb_pts_on_conic[h]);
		cout << " : ";
		cout << endl;
	}

	for (u = 0; u < 55; u++) {
		cout << "line " << u << " : ";
		int str[55];

		Orbiter->Int_vec.zero(str, 55);
		for (v = 0; v < nb_pts; v++) {
			pt = Pts[v];
			if (Sorting.lint_vec_search_linear(pts_on_line + u * nb_pts_per_line, nb_pts_per_line, pt, idx)) {
				str[v] = 1;
			}
		}
		Orbiter->Int_vec.print_fully(cout, str, 55);
		cout << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::conic_type done" << endl;
	}

}

void projective_space_with_action::cheat_sheet(
		layered_graph_draw_options *O,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::cheat_sheet" << endl;
	}



	{
		char fname[1000];
		char title[1000];
		char author[1000];

		snprintf(fname, 1000, "PG_%d_%d.tex", n, F->q);
		snprintf(title, 1000, "Cheat Sheet ${\\rm PG}(%d,%d)$", n, F->q);
		//strcpy(author, "");
		author[0] = 0;


		{
			ofstream ost(fname);
			latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					NULL /* extra_praeamble */);


			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG before A->report" << endl;
			}

			A->report(ost, A->f_has_sims, A->Sims,
					A->f_has_strong_generators, A->Strong_gens,
					O,
					verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG after PA->A->report" << endl;
			}

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG before PA->P->report" << endl;
			}



			P->report(ost, O, verbose_level);

			if (f_v) {
				cout << "projective_space_with_action::do_cheat_sheet_PG after PP->report" << endl;
			}


			L.foot(ost);

		}
		file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}

	}

	if (f_v) {
		cout << "projective_space_with_action::cheat_sheet done" << endl;
	}


}


void projective_space_with_action::do_spread_classify(int k,
		poset_classification_control *Control,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify" << endl;
	}
	spread_classify *SC;

	SC = NEW_OBJECT(spread_classify);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify before SC->init" << endl;
	}

	SC->init(
			this,
			k,
			TRUE /* f_recoordinatize */,
			verbose_level - 1);
	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify after SC->init" << endl;
	}

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify before SC->init2" << endl;
	}
	SC->init2(Control, verbose_level);
	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify after SC->init2" << endl;
	}


	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify before SC->compute" << endl;
	}

	SC->compute(verbose_level);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify after SC->compute" << endl;
	}


	FREE_OBJECT(SC);

	if (f_v) {
		cout << "projective_space_with_action::do_spread_classify done" << endl;
	}
}

void projective_space_with_action::setup_surface_with_action(
		surface_with_action *&Surf_A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "projective_space_with_action::setup_surface_with_action" << endl;
		cout << "projective_space_with_action::setup_surface_with_action verbose_level=" << verbose_level << endl;
	}


	surface_domain *Surf;


	if (f_v) {
		cout << "projective_space_with_action::setup_surface_with_action before Surf->init" << endl;
	}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "projective_space_with_action::setup_surface_with_action after Surf->init" << endl;
	}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "projective_space_with_action::setup_surface_with_action before Surf_A->init" << endl;
	}
	Surf_A->init(Surf, this, TRUE /* f_recoordinatize */, 0 /*verbose_level*/);
	if (f_v) {
		cout << "projective_space_with_action::setup_surface_with_action after Surf_A->init" << endl;
	}

}




// #############################################################################
// globals:
// #############################################################################



void OiPA_encode(void *extra_data,
		long int *&encoding, int &encoding_sz, void *global_data)
{
	//cout << "OiPA_encode" << endl;
	object_in_projective_space_with_action *OiPA;
	object_in_projective_space *OiP;

	OiPA = (object_in_projective_space_with_action *) extra_data;
	OiP = OiPA->OiP;
	//OiP->print(cout);
	OiP->encode_object(encoding, encoding_sz, 1 /* verbose_level*/);
	//cout << "OiPA_encode done" << endl;

}

void OiPA_group_order(void *extra_data,
		longinteger_object &go, void *global_data)
{
	//cout << "OiPA_group_order" << endl;
	object_in_projective_space_with_action *OiPA;
	//object_in_projective_space *OiP;

	OiPA = (object_in_projective_space_with_action *) extra_data;
	//OiP = OiPA->OiP;
	go.create(OiPA->ago, __FILE__, __LINE__);
	//OiPA->Aut_gens->group_order(go);
	//cout << "OiPA_group_order done" << endl;

}

void print_summary_table_entry(int *Table,
		int m, int n, int i, int j, int val, std::string &output, void *data)
{
	classify_bitvectors *CB;
	object_in_projective_space_with_action *OiPA;
	void *extra_data;
	longinteger_object go;
	int h;
	sorting Sorting;
	char str[1000];

	CB = (classify_bitvectors *) data;

	str[0] = 0;

	if (i == -1) {
		if (j == -1) {
			sprintf(str, "\\mbox{Orbit}");
		}
		else if (j == 0) {
			sprintf(str, "\\mbox{Rep}");
		}
		else if (j == 1) {
			sprintf(str, "\\#");
		}
		else if (j == 2) {
			sprintf(str, "\\mbox{Ago}");
		}
		else if (j == 3) {
			sprintf(str, "\\mbox{Objects}");
		}
	}
	else {
		//cout << "print_summary_table_entry i=" << i << " j=" << j << endl;
		if (j == -1) {
			sprintf(str, "%d", i);
		}
		else if (j == 2) {
			extra_data = CB->Type_extra_data[CB->perm[i]];

			OiPA = (object_in_projective_space_with_action *) extra_data;
			go.create(OiPA->ago, __FILE__, __LINE__);
			//OiPA->Aut_gens->group_order(go);
			go.print_to_string(str);
		}
		else if (j == 3) {


			int *Input_objects;
			int nb_input_objects;
			CB->C_type_of->get_class_by_value(Input_objects,
				nb_input_objects, CB->perm[i], 0 /*verbose_level */);
			Sorting.int_vec_heapsort(Input_objects, nb_input_objects);

			output[0] = 0;
			for (h = 0; h < nb_input_objects; h++) {
				sprintf(str + strlen(str), "%d", Input_objects[h]);
				if (h < nb_input_objects - 1) {
					strcat(str, ", ");
				}
				if (h == 10) {
					strcat(str, "\\ldots");
					break;
				}
			}

			FREE_int(Input_objects);
		}
		else {
			sprintf(str, "%d", val);
		}
	}
	output.assign(str);
}


void compute_ago_distribution(
	classify_bitvectors *CB, tally *&C_ago, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_ago_distribution" << endl;
	}
	long int *Ago;
	int i;

	Ago = NEW_lint(CB->nb_types);
	for (i = 0; i < CB->nb_types; i++) {
		object_in_projective_space_with_action *OiPA;

		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[i];
		Ago[i] = OiPA->ago; //OiPA->Aut_gens->group_order_as_lint();
	}
	C_ago = NEW_OBJECT(tally);
	C_ago->init_lint(Ago, CB->nb_types, FALSE, 0);
	FREE_lint(Ago);
	if (f_v) {
		cout << "compute_ago_distribution done" << endl;
	}
}

void compute_ago_distribution_permuted(
	classify_bitvectors *CB, tally *&C_ago, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_ago_distribution_permuted" << endl;
	}
	long int *Ago;
	int i;

	Ago = NEW_lint(CB->nb_types);
	for (i = 0; i < CB->nb_types; i++) {
		object_in_projective_space_with_action *OiPA;

		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[CB->perm[i]];
		Ago[i] = OiPA->ago; //OiPA->Aut_gens->group_order_as_lint();
	}
	C_ago = NEW_OBJECT(tally);
	C_ago->init_lint(Ago, CB->nb_types, FALSE, 0);
	FREE_lint(Ago);
	if (f_v) {
		cout << "compute_ago_distribution_permuted done" << endl;
	}
}

void compute_and_print_ago_distribution(ostream &ost,
	classify_bitvectors *CB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_and_print_ago_distribution" << endl;
	}
	tally *C_ago;
	compute_ago_distribution(CB, C_ago, verbose_level);
	ost << "ago distribution: " << endl;
	ost << "$$" << endl;
	C_ago->print_naked_tex(ost, TRUE /* f_backwards */);
	ost << endl;
	ost << "$$" << endl;
	FREE_OBJECT(C_ago);
}

void compute_and_print_ago_distribution_with_classes(ostream &ost,
	classify_bitvectors *CB, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	latex_interface L;

	if (f_v) {
		cout << "compute_and_print_ago_distribution_with_classes" << endl;
	}
	tally *C_ago;
	compute_ago_distribution_permuted(CB, C_ago, verbose_level);
	ost << "Ago distribution: " << endl;
	ost << "$$" << endl;
	C_ago->print_naked_tex(ost, TRUE /* f_backwards */);
	ost << endl;
	ost << "$$" << endl;
	set_of_sets *SoS;
	int *types;
	int nb_types;

	SoS = C_ago->get_set_partition_and_types(types,
			nb_types, verbose_level);


	// go backwards to show large group orders first:
	for (i = SoS->nb_sets - 1; i >= 0; i--) {
		ost << "Group order $" << types[i]
			<< "$ appears for the following $" << SoS->Set_size[i]
			<< "$ classes: $" << endl;
		L.lint_set_print_tex(ost, SoS->Sets[i], SoS->Set_size[i]);
		ost << "$\\\\" << endl;
		//int_vec_print_as_matrix(ost, SoS->Sets[i],
		//SoS->Set_size[i], 10 /* width */, TRUE /* f_tex */);
		//ost << "$$" << endl;

	}

	FREE_int(types);
	FREE_OBJECT(SoS);
	FREE_OBJECT(C_ago);
}


int table_of_sets_compare_func(void *data, int i,
		void *search_object,
		void *extra_data)
{
	long int *Data = (long int *) data;
	long int *p = (long int *) extra_data;
	long int len = p[0];
	int ret;

	ret = lint_vec_compare(Data + i * len, (long int *) search_object, len);
	return ret;
}



}}
