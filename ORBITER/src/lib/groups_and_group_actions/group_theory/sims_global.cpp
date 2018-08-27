// sims_global.C
//
// Anton Betten
// December 21, 2003
// moved here from sims.C: Dec 22, 2014



#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

// global functions:

sims *create_sims_from_generators_with_target_group_order_factorized(
	action *A,
	vector_ge *gens, INT *tl, INT len, INT verbose_level)
{
	longinteger_object go;
	longinteger_domain D;

	D.multiply_up(go, tl, len);
	return create_sims_from_generators_randomized(A, 
		gens, TRUE /* f_target_go */, go, verbose_level);
}

sims *create_sims_from_generators_with_target_group_order_INT(
	action *A,
	vector_ge *gens, INT target_go, INT verbose_level)
{
	longinteger_object tgo;

	tgo.create(target_go);
	return create_sims_from_generators_with_target_group_order(
			A, gens, tgo, verbose_level);
	
}

sims *create_sims_from_generators_with_target_group_order(
	action *A,
	vector_ge *gens, longinteger_object &target_go,
	INT verbose_level)
{
	return create_sims_from_generators_randomized(A, 
		gens, TRUE /* f_target_go */, target_go, verbose_level);
}

sims *create_sims_from_generators_without_target_group_order(
	action *A,
	vector_ge *gens, INT verbose_level)
{
	longinteger_object dummy;
	
	return create_sims_from_generators_randomized(A, 
		gens, FALSE /* f_target_go */, dummy, verbose_level);
}

sims *create_sims_from_single_generator_without_target_group_order(
	action *A,
	INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	sims *S;
	vector_ge *gens;
	longinteger_object dummy;
	
	if (f_v) {
		cout << "create_sims_from_single_generator_"
				"without_target_group_order" << endl;
		}
	gens = NEW_OBJECT(vector_ge);
	gens->init_single(A, Elt);
	
	S = create_sims_from_generators_randomized(A, 
		gens, FALSE /* f_target_go */, dummy, verbose_level);

	FREE_OBJECT(gens);
	if (f_v) {
		cout << "create_sims_from_single_generator_"
				"without_target_group_order done" << endl;
		}
	return S;
}

sims *create_sims_from_generators_randomized(action *A, 
	vector_ge *gens, INT f_target_go, longinteger_object &target_go,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//init(A);
	//init_trivial_group(0);
	//freeself();
	sims *S;

	if (f_v) {
		cout << "create_sims_from_generators_randomized" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		if (f_target_go) {
			cout << "creating a group of order " << target_go << endl;
			}
		}
	
	schreier_sims *ss;

	ss = NEW_OBJECT(schreier_sims);
	
	ss->init(A, verbose_level - 1);

	//ss->interested_in_kernel(A_subaction, verbose_level - 1);
	
	if (f_target_go) {
		ss->init_target_group_order(target_go, verbose_level - 1);
		}
	
	ss->init_generators(gens, verbose_level);
	
	ss->create_group(verbose_level - 1);

	S = ss->G;
	ss->G = NULL;
	//*this = *ss->G;
	
	//ss->G->null();
	
	//cout << "create_sims_from_generators_randomized
	// before FREE_OBJECT ss" << endl;
	FREE_OBJECT(ss);
	//cout << "create_sims_from_generators_randomized
	// after FREE_OBJECT ss" << endl;

	if (f_v) {
		cout << "create_sims_from_generators_randomized "
				"done" << endl;
		}
	return S;
}

sims *create_sims_for_centralizer_of_matrix(
		action *A, INT *Mtx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	matrix_group *M;
	finite_field *F;
	INT d, q, i;
	gl_classes *C; 

	if (f_v) {
		cout << "create_sims_for_centralizer_of_matrix" << endl;
		}

	if (A->type_G != matrix_group_t) {
		cout << "create_sims_for_centralizer_of_matrix "
				"action not of type matrix_group" << endl;
		exit(1);
		}

	M = A->G.matrix_grp;
	F = M->GFq;
	q = F->q;
	d = M->n;


	if (M->C == NULL) {
		if (f_v) {
			cout << "create_sims_for_centralizer_of_matrix "
					"before M->init_gl_classes" << endl;
			}
		M->init_gl_classes(verbose_level - 2);
		}

	C = M->C;
	
	if (f_v) {
		cout << "create_sims_for_centralizer_of_matrix "
				"d = " << d << " q = " << q << endl;
		cout << "Mtx=" << endl;
		INT_matrix_print(Mtx, d, d);
		}

	//gl_classes C;
	//gl_class_rep *Reps;
	//INT nb_classes;

	//C.init(d, F, 0 /*verbose_level - 2*/);


#if 0
	C.make_classes(Reps, nb_classes, 0 /*verbose_level - 2*/);

	if (f_v) {
		cout << "create_sims_for_centralizer_of_matrix "
				"There are " << nb_classes << " conjugacy classes" << endl;
		}
	if (f_vv) {
		cout << "create_sims_for_centralizer_of_matrix "
				"The conjugacy classes are:" << endl;
		for (i = 0; i < nb_classes; i++) {
			cout << "Class " << i << ":" << endl;
			INT_matrix_print(Reps[i].type_coding.M,
					Reps[i].type_coding.m, Reps[i].type_coding.n);
			cout << "Centralizer order = "
					<< Reps[i].centralizer_order << endl;
			}
		}
#endif

	
	//INT class_rep;

	INT *Elt;

	Elt = NEW_INT(A->elt_size_in_INT);

	gl_class_rep *R1;

	R1 = NEW_OBJECT(gl_class_rep);

	INT *Basis;
	INT **Gens;
	INT nb_gens;
	INT nb_alloc = 20;
		
	Gens = NEW_PINT(nb_alloc);
	nb_gens = 0;
			
	Basis = NEW_INT(d * d);
	if (f_v) {
		cout << "create_sims_for_centralizer_of_matrix "
				"before generators_for_centralizer" << endl;
		}
	C->generators_for_centralizer(Mtx, R1, Basis, Gens,
			nb_gens, nb_alloc, verbose_level - 2);

	if (f_v) {
		cout << "create_sims_for_centralizer_of_matrix "
				"Basis=" << endl;
		INT_matrix_print(Basis, d, d);
		cout << "create_sims_for_centralizer_of_matrix "
				"We found " << nb_gens << " centralizing matrices" << endl;
		}

	if (f_vv) {
		cout << "create_sims_for_centralizer_of_matrix "
				"Gens=" << endl;
		for (i = 0; i < nb_gens; i++) {
			cout << "Gen " << i << " / " << nb_gens << " is:" << endl;
			INT_matrix_print(Gens[i], d, d);
			}
		}

	for (i = 0; i < nb_gens; i++) {
		if (!F->test_if_commute(Mtx, Gens[i], d,
				0/*verbose_level*/)) {
			cout << "The matrices do not commute" << endl;
			cout << "Mtx=" << endl;
			INT_matrix_print(Mtx, d, d);
			cout << "Gens[i]=" << endl;
			INT_matrix_print(Gens[i], d, d);
			exit(1);
			}
		}

	//C.identify_matrix(Elt, R1, verbose_level);

	if (f_v) {
		cout << "The type of the matrix under "
				"consideration is:" << endl;
		INT_matrix_print(R1->type_coding.M,
				R1->type_coding.m, R1->type_coding.n);
		}


#if 0
	class_rep = C.find_class_rep(Reps, nb_classes,
			R1, 0 /* verbose_level */);

	if (f_v) {
		cout << "The index of the class of the "
				"matrix is = " << class_rep << endl;
		}
#endif


	vector_ge *gens;
	vector_ge *SG;
	INT *tl;
	longinteger_object centralizer_order, cent_go;
	INT *Elt1;
		
	gens = NEW_OBJECT(vector_ge);
	SG = NEW_OBJECT(vector_ge);
	tl = NEW_INT(A->base_len);
	gens->init(A);
	gens->allocate(nb_gens);
	Elt1 = NEW_INT(A->elt_size_in_INT);
		
	for (i = 0; i < nb_gens; i++) {
		A->make_element(Elt1, Gens[i], 0);
		A->element_move(Elt1, gens->ith(i), 0);
		}
	sims *Cent;


	if (f_v) {
		cout << "before centralizer_order_Kung" << endl;
		}
	R1->centralizer_order_Kung(C, centralizer_order, verbose_level);
	if (f_v) {
		cout << "after centralizer_order_Kung" << endl;
		}

	Cent = create_sims_from_generators_with_target_group_order(
			A, gens,
		centralizer_order /*Reps[class_rep].centralizer_order*/,
		0 /* verbose_level */);
	//Cent = create_sims_from_generators_without_target_group_order(
	// A, gens, 0 /* verbose_level */);
	Cent->group_order(cent_go);

	if (f_v) {
		cout << "create_sims_for_centralizer_of_matrix "
				"The order of the centralizer is " << cent_go << endl;
		}




	for (i = 0; i < nb_gens; i++) {
		FREE_INT(Gens[i]);
		}
	FREE_PINT(Gens);

	FREE_OBJECT(R1);
	FREE_OBJECT(gens);
	FREE_OBJECT(SG);
	FREE_INT(tl);
	FREE_INT(Elt1);
	FREE_INT(Elt);
	FREE_INT(Basis);

	if (f_v) {
		cout << "create_sims_for_centralizer_of_matrix done" << endl;
		}
	return Cent;
}



