// gl_classes.cpp
//
// Anton Betten
// October 21, 2013

#include "orbiter.h"

using namespace std;


using namespace orbiter;

void do_GL(int q, int d, int f_no_eigenvalue_one, int verbose_level);
void do_random(int q, int d, int f_no_eigenvalue_one, int verbose_level);
void do_identify_all(int q, int d,
		int f_no_eigenvalue_one, int verbose_level);
void do_identify_one(int q, int d, int f_no_eigenvalue_one,
		int elt_idx, int verbose_level);
void do_normal_form(int q, int d, int f_no_eigenvalue_one,
		int *data, int data_sz, int verbose_level);

int main(int argc, char **argv)
{
	int t0 = os_ticks();
	int verbose_level = 0;
	int i;
	int f_GL = FALSE;
	int q, d;
	int f_no_eigenvalue_one = FALSE;
	int f_random = FALSE;
	int f_identify_all = FALSE;
	int f_identify_one = FALSE;
	int f_group_table = FALSE;
	int f_centralizer_brute_force = FALSE;
	int f_centralizer = FALSE;
	int elt_idx = -1;
	int f_centralizer_all = FALSE;
	int f_normal_form = FALSE;
	const char *normal_form_data = NULL;
	int *data = NULL;
	int data_sz = 0;
	int f_poly = FALSE;
	const char *poly = NULL;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-GL") == 0) {
			f_GL = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-GL " << d << " " << q << endl;
			}
		else if (strcmp(argv[i], "-no_eigenvalue_one") == 0) {
			f_no_eigenvalue_one = TRUE;
			cout << "-no_eigenvalue_one" << endl;
			}
		else if (strcmp(argv[i], "-random") == 0) {
			f_random = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-random " << d << " " << q << endl;
			}
		else if (strcmp(argv[i], "-identify_all") == 0) {
			f_identify_all = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-identify_all " << d << " " << q << endl;
			}
		else if (strcmp(argv[i], "-identify_one") == 0) {
			f_identify_one = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			elt_idx = atoi(argv[++i]);
			cout << "-identify_one " << d << " " << q
					<< " " << elt_idx << endl;
			}
		else if (strcmp(argv[i], "-normal_form") == 0) {
			f_normal_form = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			normal_form_data = argv[++i];
			cout << "-normal_form " << d << " " << q
					<< " " << normal_form_data << endl;
			}
		else if (strcmp(argv[i], "-group_table") == 0) {
			f_group_table = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-group_table " << d << " " << q << endl;
			}
		else if (strcmp(argv[i], "-centralizer_brute_force") == 0) {
			f_centralizer = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			elt_idx = atoi(argv[++i]);
			cout << "-centralizer_brute_force " << d << " "
					<< q << " " << elt_idx << endl;
			}
		else if (strcmp(argv[i], "-centralizer") == 0) {
			f_centralizer = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			elt_idx = atoi(argv[++i]);
			cout << "-centralizer " << d << " " << q << " "
					<< elt_idx << endl;
			}
		else if (strcmp(argv[i], "-centralizer_all") == 0) {
			f_centralizer_all = TRUE;
			d = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-centralizer_all " << d << " " << q << endl;
			}
		}
	

	if (f_GL) {

		do_GL(q, d, f_no_eigenvalue_one, verbose_level);

		}

	else if (f_random) {

		do_random(q, d, f_no_eigenvalue_one, verbose_level);

		}
	else if (f_identify_all) {


		do_identify_all(q, d, f_no_eigenvalue_one, verbose_level);

		}

	else if (f_identify_one) {

		do_identify_one(q, d, f_no_eigenvalue_one,
				elt_idx, verbose_level);

		}

	else if (f_normal_form) {

		int_vec_scan(normal_form_data, data, data_sz);
		if (data_sz != d * d) {
			cout << "data_sz != d * d" << endl;
			exit(1);
			}
		do_normal_form(q, d, f_no_eigenvalue_one,
				data, data_sz, verbose_level);

		}

	else if (f_group_table) {
		gl_classes C;
		gl_class_rep *Reps;
		int nb_classes;
		int *Class_rep;
		int *List;
		int list_sz, a, b, j, h;
		finite_field *F;
		
		F = new finite_field;
		if (f_poly) {
			F->init_override_polynomial(q, poly, 0);
			}
		else {
			F->init(q, 0);
			}

		C.init(d, F, verbose_level);

		C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);


		action *A;
		longinteger_object Go;
		vector_ge *nice_gens;
	
		
		A = new action;
		A->init_projective_group(d /* n */,
				F,
				FALSE /* f_semilinear */,
				TRUE /* f_basis */,
				nice_gens,
				verbose_level);
		FREE_OBJECT(nice_gens);
		A->print_base();
		A->group_order(Go);
	

		int i, go, class_rep;
		int eval;

		int *Elt;
		int *Elt1;
		int *Elt2;
		int *Elt3;
		int *Basis;

		Elt = NEW_int(A->elt_size_in_int);
		Elt1 = NEW_int(A->elt_size_in_int);
		Elt2 = NEW_int(A->elt_size_in_int);
		Elt3 = NEW_int(A->elt_size_in_int);
		Basis = NEW_int(d * d);




		go = Go.as_int();
		List = NEW_int(go);
		list_sz = 0;
		for (i = 0; i < go; i++) {

			cout << "Looking at element " << i << ":" << endl;

			A->Sims->element_unrank_int(i, Elt);
			int_matrix_print(Elt, d, d);

			{
			unipoly_domain U(C.F);
			unipoly_object char_poly;



			U.create_object_by_rank(char_poly, 0);
		
			U.characteristic_polynomial(Elt,
					d, char_poly, verbose_level - 2);

			cout << "The characteristic polynomial is ";
			U.print_object(char_poly, cout);
			cout << endl;

			eval = U.substitute_scalar_in_polynomial(char_poly,
					1 /* scalar */, 0 /* verbose_level */);
			U.delete_object(char_poly);


			}

			if (eval) {
				List[list_sz++] = i;
				}

			} // next i

		cout << "Found " << list_sz
				<< " elements without eigenvalue one" << endl;

		
		Class_rep = NEW_int(list_sz);

		for (i = 0; i < list_sz; i++) {
			a = List[i];
			
			cout << "Looking at element " << a << ":" << endl;

			A->Sims->element_unrank_int(a, Elt);
			int_matrix_print(Elt, d, d);


			gl_class_rep *R1;

			R1 = new gl_class_rep;
		
			C.identify_matrix(Elt, R1, Basis, verbose_level);

			class_rep = C.find_class_rep(Reps,
					nb_classes, R1, 0 /* verbose_level */);


			delete R1;


			cout << "class = " << class_rep << endl;
			Class_rep[i] = class_rep;
			}	

		int *Group_table;
		int *Table;
		
		Group_table = NEW_int(list_sz * list_sz);
		int_vec_zero(Group_table, list_sz * list_sz);
		for (i = 0; i < list_sz; i++) {
			a = List[i];
			A->Sims->element_unrank_int(a, Elt1);
			for (j = 0; j < list_sz; j++) {
				b = List[j];
				A->Sims->element_unrank_int(b, Elt2);
				A->element_mult(Elt1, Elt2, Elt3, 0);
				h = A->Sims->element_rank_int(Elt3);
				Group_table[i * list_sz + j] = h;
				}
			}
		int L_sz = list_sz + 1;
		Table = NEW_int(L_sz * L_sz);
		int_vec_zero(Table, L_sz * L_sz);
		for (i = 0; i < list_sz; i++) {
			Table[0 * L_sz + 1 + i] = List[i];
			Table[(i + 1) * L_sz + 0] = List[i];
			}
		for (i = 0; i < list_sz; i++) {
			for (j = 0; j < list_sz; j++) {
				Table[(i + 1) * L_sz + 1 + j] =
						Group_table[i * list_sz + j];
				}
			}
		cout << "extended group table:" << endl;
		int_matrix_print(Table, L_sz, L_sz);


		const char *fname = "group_table.tex";

		{
		ofstream fp(fname);
		latex_interface L;

		L.head(fp, FALSE /* f_book */, FALSE /* f_title */,
			"" /*const char *title */, "" /*const char *author */, 
			FALSE /* f_toc */, FALSE /* f_landscape */, FALSE /* f_12pt */, 
			FALSE /* f_enlarged_page */, FALSE /* f_pagenumbers */, 
			NULL /* extra_praeamble */);
		

		print_integer_matrix_tex_block_by_block(fp, Table, L_sz, L_sz, 15);


		
		L.foot(fp);

		}


		FREE_int(Elt);
		FREE_int(Elt1);
		FREE_int(Elt2);
		FREE_int(Elt3);
		FREE_int(Basis);
		delete A;
		delete F;
		delete [] Reps;

		}
	
	else if (f_centralizer_brute_force) {
		action *A;
		longinteger_object Go;
		finite_field *F;
		vector_ge *nice_gens;

		F = new finite_field;
		F->init(q, 0);
		
		A = new action;
		A->init_projective_group(d /* n */, F,
				FALSE /* f_semilinear */,
				TRUE /* f_basis */,
				nice_gens,
				verbose_level);
		FREE_OBJECT(nice_gens);
		A->print_base();
		A->group_order(Go);
	

		int i, go;

		int *Elt;
		int *Eltv;
		int *Elt1;
		int *Elt2;
		int *Elt3;
		int *List;
		int sz;

		Elt = NEW_int(A->elt_size_in_int);
		Eltv = NEW_int(A->elt_size_in_int);
		Elt1 = NEW_int(A->elt_size_in_int);
		Elt2 = NEW_int(A->elt_size_in_int);
		Elt3 = NEW_int(A->elt_size_in_int);




		go = Go.as_int();
		List = NEW_int(go);
		sz = 0;



		A->Sims->element_unrank_int(elt_idx, Elt);

		cout << "Computing centralizer of element "
				<< elt_idx << ":" << endl;
		int_matrix_print(Elt, d, d);

		A->element_invert(Elt, Eltv, 0);

		for (i = 0; i < go; i++) {

			cout << "Looking at element " << i << " / " << go << endl;

			A->Sims->element_unrank_int(i, Elt1);
			//int_matrix_print(Elt1, d, d);


			A->element_invert(Elt1, Elt2, 0);
			A->element_mult(Elt2, Elt, Elt3, 0);
			A->element_mult(Elt3, Elt1, Elt2, 0);
			A->element_mult(Elt2, Eltv, Elt3, 0);
			if (A->is_one(Elt3)) {
				List[sz++] = i;
				}
			}

		cout << "The centralizer has order " << sz << endl;

		int a;
		vector_ge *gens;
		vector_ge *SG;
		int *tl;
		
		gens = new vector_ge;
		SG = new vector_ge;
		tl = NEW_int(A->Stabilizer_chain->base_len);
		gens->init(A);
		gens->allocate(sz);
		
		for (i = 0; i < sz; i++) {
			a = List[i];

			cout << "Looking at element " << i << " / " << sz
					<< " which is " << a << endl;

			A->Sims->element_unrank_int(a, Elt1);
			int_matrix_print(Elt1, d, d);
			
			A->element_move(Elt1, gens->ith(i), 0);
			}

		sims *Cent;

		Cent = A->create_sims_from_generators_with_target_group_order_int(
				gens, sz, 0 /* verbose_level */);
		Cent->extract_strong_generators_in_order(*SG, tl,
				0 /* verbose_level */);
		cout << "strong generators for the centralizer are:" << endl;
		for (i = 0; i < SG->len; i++) {

			A->element_move(SG->ith(i), Elt1, 0);
			a = A->Sims->element_rank_int(Elt1);

			cout << "Element " << i << " / " << SG->len
					<< " which is " << a << endl;

			int_matrix_print(Elt1, d, d);
			
			}

		
		
		FREE_int(Elt);
		FREE_int(Eltv);
		FREE_int(Elt1);
		FREE_int(Elt2);
		FREE_int(Elt3);
		delete A;
		delete F;
		}
	else if (f_centralizer) {

		finite_field *F;
		action *A_PGL;
		action *A_GL;
		longinteger_object Go;
		vector_ge *nice_gens;
	
		F = new finite_field;
		F->init(q, 0);
		
		A_PGL = new action;
		A_PGL->init_projective_group(d /* n */, F, 
			FALSE /* f_semilinear */,
			TRUE /* f_basis */,
			nice_gens,
			0 /*verbose_level*/);
		FREE_OBJECT(nice_gens);
		A_PGL->print_base();
		A_PGL->group_order(Go);

		A_GL = new action;
		A_GL->init_general_linear_group(d /* n */, F, 
			FALSE /* f_semilinear */, 
			TRUE /* f_basis */,
			nice_gens,
			0 /*verbose_level*/);
		FREE_OBJECT(nice_gens);
		A_GL->print_base();
		A_GL->group_order(Go);

		int *Elt;

		Elt = NEW_int(A_PGL->elt_size_in_int);


		//go = Go.as_int();

		cout << "Looking at element " << elt_idx << ":" << endl;

		A_PGL->Sims->element_unrank_int(elt_idx, Elt);
		int_matrix_print(Elt, d, d);

		strong_generators *Cent;
		strong_generators *Cent_GL;
		longinteger_object go, go1;

		Cent = new strong_generators;
		Cent_GL = new strong_generators;
		
		cout << "before Cent->init_centralizer_of_matrix" << endl;
		Cent->init_centralizer_of_matrix(A_PGL, Elt, verbose_level);
		cout << "before Cent->init_centralizer_of_matrix" << endl;

		cout << "before Cent_GL->init_centralizer_of_matrix_general_linear" << endl;
		Cent_GL->init_centralizer_of_matrix_general_linear(
				A_PGL, A_GL, Elt, verbose_level);
		cout << "after Cent_GL->init_centralizer_of_matrix_general_linear" << endl;
	

		
		Cent->group_order(go);
		Cent_GL->group_order(go1);
		
		cout << "order of centralizer in PGL: " << go << " in GL: " << go1 << endl;
		FREE_int(Elt);
		delete Cent;
		delete Cent_GL;
		delete A_PGL;
		delete A_GL;
		delete F;
		}

	else if (f_centralizer_all) {

		action *A;
		finite_field *F;
		longinteger_object Go;
		vector_ge *nice_gens;
		int go, i;
	
		F = new finite_field;
		F->init(q, 0);
		A = new action;
		A->init_projective_group(d /* n */, F,
				FALSE /* f_semilinear */,
				TRUE /* f_basis */,
				nice_gens,
				0 /*verbose_level*/);
		FREE_OBJECT(nice_gens);
		A->print_base();
		A->group_order(Go);

		int *Elt;

		Elt = NEW_int(A->elt_size_in_int);


		go = Go.as_int();

		for (i = 0; i < go; i++) {
			cout << "Looking at element " << i << ":" << endl;

			A->Sims->element_unrank_int(i, Elt);
			int_matrix_print(Elt, d, d);

			sims *Cent;
			longinteger_object cent_go;

			Cent = A->create_sims_for_centralizer_of_matrix(
					Elt, verbose_level);
			Cent->group_order(cent_go);

			cout << "Looking at element " << i
					<< ", the centralizer has order " << cent_go << endl;
			
			

			delete Cent;

			}



		FREE_int(Elt);
		delete A;
		delete F;
		}


	
	time_check(cout, t0);
	cout << endl;
}


void do_GL(int q, int d, int f_no_eigenvalue_one, int verbose_level)
{
	gl_classes C;
	gl_class_rep *R;
	int nb_classes;
	finite_field *F;
	int i;
		
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	C.init(d, F, verbose_level);

	C.make_classes(R, nb_classes, f_no_eigenvalue_one, verbose_level);

	action *A;
	longinteger_object Go;
	vector_ge *nice_gens;
	int a;
	int *Mtx;
	int *Elt;
	
		

	A = NEW_OBJECT(action);
	A->init_projective_group(d /* n */, F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);
	
	Mtx = NEW_int(d * d);
	Elt = NEW_int(A->elt_size_in_int);


	for (i = 0; i < nb_classes; i++) {
			
		C.make_matrix_from_class_rep(Mtx, R + i, 0 /*verbose_level - 1 */);

		A->make_element(Elt, Mtx, 0);
			
		a = A->Sims->element_rank_int(Elt);

		cout << "Representative of class " << i << " / "
				<< nb_classes << " has rank " << a << endl;
		int_matrix_print(Elt, d, d);

		C.print_matrix_and_centralizer_order_latex(
				cout, R + i);

		}


	char fname[1000];

	sprintf(fname, "Class_reps_GL_%d_%d.tex", d, q);
	{
	ofstream fp(fname);
	latex_interface L;

	L.head_easy(fp);
	C.report(fp, verbose_level);
	L.foot(fp);
	}

	//make_gl_classes(d, q, f_no_eigenvalue_one, verbose_level);

	FREE_int(Mtx);
	FREE_int(Elt);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
	delete [] R;
}

void do_random(int q, int d, int f_no_eigenvalue_one, int verbose_level)
{
	//gl_random_matrix(d, q, verbose_level);

	gl_classes C;
	gl_class_rep *Reps;
	int nb_classes;
	finite_field *F;
		
	F = new finite_field;
	F->init(q, 0);
	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);

	int *Mtx;
	int *Basis;
	int class_rep;


	Mtx = NEW_int(d * d);
	Basis = NEW_int(d * d);

	C.F->random_invertible_matrix(Mtx, d, verbose_level - 2);

	
	gl_class_rep *R1;

	R1 = new gl_class_rep;
		
	C.identify_matrix(Mtx, R1, Basis, verbose_level);

	class_rep = C.find_class_rep(Reps, nb_classes,
			R1, 0 /* verbose_level */);

	cout << "class = " << class_rep << endl;
		
	delete F;
	delete R1;

	FREE_int(Mtx);
	FREE_int(Basis);
	delete [] Reps;
}

void do_identify_all(int q, int d,
		int f_no_eigenvalue_one, int verbose_level)
{
	gl_classes C;
	gl_class_rep *Reps;
	int nb_classes;
	finite_field *F;
		
	F = new finite_field;
	F->init(q, 0);

	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);



	action *A;
	longinteger_object Go;
	int *Class_count;
	vector_ge *nice_gens;
	
		
	A = new action;
	A->init_projective_group(d /* n */, F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);
	

	int i, go, class_rep;

	int *Elt, *Basis;

	Class_count = NEW_int(nb_classes);
	int_vec_zero(Class_count, nb_classes);
	Elt = NEW_int(A->elt_size_in_int);
	Basis = NEW_int(d * d);

	go = Go.as_int();
	for (i = 0; i < go; i++) {

		cout << "Looking at element " << i << ":" << endl;

		A->Sims->element_unrank_int(i, Elt);
		int_matrix_print(Elt, d, d);


		gl_class_rep *R1;

		R1 = new gl_class_rep;
		
		C.identify_matrix(Elt, R1, Basis, verbose_level);

		class_rep = C.find_class_rep(Reps,
				nb_classes, R1, 0 /* verbose_level */);

		cout << "class = " << class_rep << endl;

		Class_count[class_rep]++;

		delete R1;
		}

	cout << "class : count" << endl;
	for (i = 0; i < nb_classes; i++) {
		cout << setw(3) << i << " : " << setw(10)
				<< Class_count[i] << endl;
		}

		

	FREE_int(Class_count);
	FREE_int(Elt);
	FREE_int(Basis);
	delete A;
	delete F;
	delete [] Reps;
}

void do_identify_one(int q, int d,
		int f_no_eigenvalue_one, int elt_idx,
		int verbose_level)
{
	gl_classes C;
	gl_class_rep *Reps;
	int nb_classes;
	finite_field *F;
		
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	C.init(d, F, verbose_level);

	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one, verbose_level);



	action *A;
	longinteger_object Go;
	vector_ge *nice_gens;
	
		
	A = NEW_OBJECT(action);
	A->init_projective_group(d /* n */, F,
			FALSE /* f_semilinear */,
			TRUE /* f_basis */,
			nice_gens,
			verbose_level);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);
	

	int class_rep;

	int *Elt, *Basis;

	Elt = NEW_int(A->elt_size_in_int);
	Basis = NEW_int(d * d);

	//int go;
	//go = Go.as_int();

	cout << "Looking at element " << elt_idx << ":" << endl;

	A->Sims->element_unrank_int(elt_idx, Elt);
	int_matrix_print(Elt, d, d);


	gl_class_rep *R1;

	R1 = NEW_OBJECT(gl_class_rep);
		
	C.identify_matrix(Elt, R1, Basis, verbose_level);

	class_rep = C.find_class_rep(Reps, nb_classes, R1, 0 /* verbose_level */);

	cout << "class = " << class_rep << endl;

	FREE_OBJECT(R1);


		

	FREE_int(Elt);
	FREE_int(Basis);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
	delete [] Reps;
}

void do_normal_form(int q, int d,
		int f_no_eigenvalue_one, int *data, int data_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	gl_classes C;
	gl_class_rep *Reps;
	int nb_classes;
	finite_field *F;
		
	if (f_v) {
		cout << "do_normal_form" << endl;
		}
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	if (f_v) {
		cout << "do_normal_form before C.init" << endl;
		}
	C.init(d, F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "do_normal_form after C.init" << endl;
		}

	if (f_v) {
		cout << "do_normal_form before C.make_classes" << endl;
		}
	C.make_classes(Reps, nb_classes, f_no_eigenvalue_one,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "do_normal_form after C.make_classes" << endl;
		}



	action *A;
	longinteger_object Go;
	vector_ge *nice_gens;
	
		
	A = NEW_OBJECT(action);
	A->init_projective_group(d /* n */, F,
			FALSE /* f_semilinear */, TRUE /* f_basis */,
			nice_gens,
			0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);
	A->print_base();
	A->group_order(Go);
	

	int class_rep;

	int *Elt, *Basis;

	Elt = NEW_int(A->elt_size_in_int);
	Basis = NEW_int(d * d);

	//go = Go.as_int();

	cout << "Making element from data ";
	int_vec_print(cout, data, data_sz);
	cout << endl;

	//A->Sims->element_unrank_int(elt_idx, Elt);
	A->make_element(Elt, data, verbose_level);

	cout << "Looking at element:" << endl;
	int_matrix_print(Elt, d, d);


	gl_class_rep *R1;

	R1 = NEW_OBJECT(gl_class_rep);
		
	C.identify_matrix(Elt, R1, Basis, verbose_level);

	class_rep = C.find_class_rep(Reps, nb_classes, R1,
			0 /* verbose_level */);

	cout << "class = " << class_rep << endl;

	FREE_OBJECT(R1);


		

	FREE_int(Elt);
	FREE_int(Basis);
	FREE_OBJECT(A);
	FREE_OBJECT(F);
	FREE_OBJECTS(Reps);
}


