
// wreath_product.C
//
// Anton Betten
//
// August 4, 2018
//
//
//

#include "orbiter.h"

using namespace orbiter;



// global data:

int t0; // the system time when the program started

void usage(int argc, const char **argv);
int main(int argc, const char **argv);
int wreath_rank_point_func(int *v, void *data);
void wreath_unrank_point_func(int *v, int rk, void *data);
void wreath_product_print_set(ostream &ost, int len, int *S, void *data);



typedef class tensor_product tensor_product;

//! classification of tensors under the wreath product group


class tensor_product {
public:
	int argc;
	const char **argv;
	int nb_factors;
	int n;
	int q;

	finite_field *F;
	action *A;
	action *A0;

	strong_generators *SG;
	longinteger_object go;
	wreath_product *W;
	vector_space *VS;
	poset *Poset;
	poset_classification *Gen;
	int vector_space_dimension;
	int *v; // [vector_space_dimension]

	tensor_product();
	~tensor_product();
	void init(int argc, const char **argv,
			int nb_factors, int n, int q, int depth,
			int verbose_level);
};





void usage(int argc, const char **argv)
{
	cout << "usage: " << argv[0] << " [options]" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <n>                   : verbose level n" << endl;
	cout << "-nb_factors <nb_factors> : set number of factors" << endl;
	cout << "-d <d>                   : set dimension d" << endl;
	cout << "-q <q>                   : set field size q" << endl;
}

/*-------------------------------------------------------*/
// CUDA Stuff
/*-------------------------------------------------------*/
#ifdef __CUDA_ARCH__
#include "CUDA/Linalg.h"
#endif
/*-------------------------------------------------------*/


int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;
	int f_nb_factors = FALSE;
	int nb_factors = 0;
	int f_d = FALSE;
	int d = 0;
	int f_q = FALSE;
	int q = 0;
	int f_depth = FALSE;
	int depth = 0;

	t0 = os_ticks();

	//f_memory_debug = TRUE;
	//f_memory_debug_verbose = TRUE;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-h") == 0) {
			usage(argc, argv);
			exit(1);
			}
		else if (strcmp(argv[i], "-help") == 0) {
			usage(argc, argv);
			exit(1);
			}
		else if (strcmp(argv[i], "-nb_factors") == 0) {
			f_nb_factors = TRUE;
			nb_factors = atoi(argv[++i]);
			cout << "-nb_factors " << nb_factors << endl;
			}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
			}
		}
	if (!f_nb_factors) {
		cout << "please use -nb_factors <nb_factors>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_d) {
		cout << "please use -d <d>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q <q>" << endl;
		usage(argc, argv);
		exit(1);
		}
	if (!f_depth) {
		cout << "please use -depth <depth>" << endl;
		usage(argc, argv);
		exit(1);
		}


	//do_it(argc, argv, nb_factors, d, q, verbose_level);


	tensor_product *T;

	T = NEW_OBJECT(tensor_product);

	T->init(argc, argv, nb_factors, d, q, depth, verbose_level);

	the_end_quietly(t0);

}

tensor_product::tensor_product()
{
	argc= 0;
	argv = NULL;
	nb_factors = 0;
	vector_space_dimension = 0;
	v = NULL;
	n = 0;
	q = 0;
	SG = NULL;
	F = NULL;
	A = NULL;
	A0 = NULL;
	W = NULL;
	VS = NULL;
	Poset = NULL;
	Gen = NULL;
}

tensor_product::~tensor_product()
{

}

void tensor_product::init(int argc, const char **argv,
		int nb_factors, int n, int q, int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int *v;
	int i, j, a;

	if (f_v) {
		cout << "tensor_product::init" << endl;
	}
	tensor_product::argc = argc;
	tensor_product::argv = argv;
	tensor_product::nb_factors = nb_factors;
	tensor_product::n = n;
	tensor_product::q = q;

	A = NEW_OBJECT(action);

	//v = NEW_int(n);


	F = NEW_OBJECT(finite_field);

	F->init(q, 0);

	A->init_wreath_product_group_and_restrict(nb_factors, n,
			F,
			verbose_level);
	cout << "tensor_product::init after "
			"A->init_wreath_product_group_and_restrict" << endl;

	if (!A->f_has_subaction) {
		cout << "tensor_product::init action "
				"A does not have a subaction" << endl;
		exit(1);
	}
	A0 = A->subaction;

	W = A0->G.wreath_product_group;

	vector_space_dimension = W->dimension_of_tensor_action;

	if (!A0->f_has_strong_generators) {
		cout << "tensor_product::init action A0 does not "
				"have strong generators" << endl;
		exit(1);
		}

	v = NEW_int(vector_space_dimension);

	SG = A0->Strong_gens;
	SG->group_order(go);

	cout << "tensor_product::init The group " << A->label
			<< " has order " << go
			<< " and permutation degree " << A->degree << endl;


#if 0
	i = SG->gens->len - 1;
	cout << "generator " << i << " is: " << endl;


	int h;

	cout << "computing image of 2:" << endl;
	h = A->element_image_of(2,
			SG->gens->ith(i), 10 /*verbose_level - 2*/);


	for (j = 0; j < A->degree; j++) {
		h = A->element_image_of(j,
				SG->gens->ith(i), verbose_level - 2);
		cout << j << " -> " << h << endl;
	}

		A->element_print_as_permutation(SG->gens->ith(i), cout);
	cout << endl;
#endif

	cout << "tensor_product::init Generators are:" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		cout << "generator " << i << " / "
				<< SG->gens->len << " is: " << endl;
		A->element_print_quick(SG->gens->ith(i), cout);
		cout << "as permutation: " << endl;
		if (A->degree < 200) {
			A->element_print_as_permutation_with_offset(
					SG->gens->ith(i), cout,
					0 /* offset*/,
					TRUE /* f_do_it_anyway_even_for_big_degree*/,
					TRUE /* f_print_cycles_of_length_one*/,
					0 /* verbose_level*/);
			//A->element_print_as_permutation(SG->gens->ith(i), cout);
			cout << endl;
		} else {
			cout << "too big to print" << endl;
		}
	}
	cout << "tensor_product::init Generators as permutations are:" << endl;



	if (A->degree < 200) {
		for (i = 0; i < SG->gens->len; i++) {
			A->element_print_as_permutation(SG->gens->ith(i), cout);
			cout << endl;
		}
	}
	else {
		cout << "too big to print" << endl;
	}
	cout << "tensor_product::init Generators in GAP format are:" << endl;
	if (A->degree < 200) {
		cout << "G := Group([";
		for (i = 0; i < SG->gens->len; i++) {
			A->element_print_as_permutation_with_offset(
					SG->gens->ith(i), cout,
					1 /*offset*/,
					TRUE /* f_do_it_anyway_even_for_big_degree */,
					FALSE /* f_print_cycles_of_length_one */,
					0 /* verbose_level*/);
			if (i < SG->gens->len - 1) {
				cout << ", " << endl;
			}
		}
		cout << "]);" << endl;
	}
	else {
		cout << "too big to print" << endl;
	}
	cout << "tensor_product::init "
			"Generators in compact permutation form are:" << endl;
	if (A->degree < 200) {
		cout << SG->gens->len << " " << A->degree << endl;
		for (i = 0; i < SG->gens->len; i++) {
			for (j = 0; j < A->degree; j++) {
				a = A->element_image_of(j,
						SG->gens->ith(i), 0 /* verbose_level */);
				cout << a << " ";
				}
			cout << endl;
			}
		cout << "-1" << endl;
	}
	else {
		cout << "too big to print" << endl;
	}



#ifdef __CUDA_ARCH__
#include "CUDA/Matrix.h"

	const size_t __matrix__size__ = 50000;

	Matrix<int> M(__matrix__size__, __matrix__size__),
				N(__matrix__size__, __matrix__size__),
				P(__matrix__size__, __matrix__size__);

	for (size_t i=0; i<5; ++i)
		for (size_t j=0; j<5; ++j)
			M(i,j) = N(i,j) + i+j;

	cout << "M Before:"; M.nrows = M.ncols = 5; M.print();M.nrows = M.ncols = __matrix__size__; cout << endl;
	cout << "N Before:"; N.nrows = N.ncols = 5; N.print();N.nrows = N.ncols = __matrix__size__; cout << endl;
	cout << "P Before:"; P.nrows = P.ncols = 5; P.print();P.nrows = P.ncols = __matrix__size__; cout << endl;

	linalg::device_dot(M, N, P);

	cout << "P Before:"; P.nrows = P.ncols = 5; P.print();P.nrows = P.ncols = __matrix__size__; cout << endl;

#endif




#if 0

	cout << "tensor_product::init testing..." << endl;
	int r1, r2;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;
	int *perm1;
	int *perm2;
	int *perm3;
	int *perm4;
	int *perm5;
	int cnt;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	perm1 = NEW_int(A->degree);
	perm2 = NEW_int(A->degree);
	perm3 = NEW_int(A->degree);
	perm4 = NEW_int(A->degree);
	perm5 = NEW_int(A->degree);

	for (cnt = 0; cnt < 10; cnt++) {
		r1 = random_integer(SG->gens->len);
		r2 = random_integer(SG->gens->len);
		cout << "r1=" << r1 << endl;
		cout << "r2=" << r2 << endl;
		A->element_move(SG->gens->ith(r1), Elt1, 0);
		A->element_move(SG->gens->ith(r2), Elt2, 0);
		cout << "Elt1 = " << endl;
		A->element_print_quick(Elt1, cout);
		A->element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
		cout << "as permutation: " << endl;
		perm_print(cout, perm1, A->degree);
		cout << endl;

		cout << "Elt2 = " << endl;
		A->element_print_quick(Elt2, cout);
		A->element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
		cout << "as permutation: " << endl;
		perm_print(cout, perm2, A->degree);
		cout << endl;

		A->element_mult(Elt1, Elt2, Elt3, 0);
		cout << "Elt3 = " << endl;
		A->element_print_quick(Elt3, cout);
		A->element_as_permutation(Elt3, perm3, 0 /* verbose_level */);
		cout << "as permutation: " << endl;
		perm_print(cout, perm3, A->degree);
		cout << endl;

		perm_mult(perm1, perm2, perm4, A->degree);
		cout << "perm1 * perm2= " << endl;
		perm_print(cout, perm4, A->degree);
		cout << endl;

		for (i = 0; i < A->degree; i++) {
			if (perm3[i] != perm4[i]) {
				cout << "test " << cnt
						<< " failed; something is wrong" << endl;
				exit(1);
			}
		}
	}
	cout << "tensor_product::init test 1 passed" << endl;


	for (cnt = 0; cnt < 10; cnt++) {
		r1 = random_integer(SG->gens->len);
		cout << "r1=" << r1 << endl;
		A->element_move(SG->gens->ith(r1), Elt1, 0);
		cout << "Elt1 = " << endl;
		A->element_print_quick(Elt1, cout);
		A->element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
		cout << "as permutation: " << endl;
		perm_print(cout, perm1, A->degree);
		cout << endl;

		A->element_invert(Elt1, Elt2, 0);
		cout << "Elt2 = " << endl;
		A->element_print_quick(Elt2, cout);
		A->element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
		cout << "as permutation: " << endl;
		perm_print(cout, perm2, A->degree);
		cout << endl;

		A->element_mult(Elt1, Elt2, Elt3, 0);
		cout << "Elt3 = " << endl;
		A->element_print_quick(Elt3, cout);
		A->element_as_permutation(Elt3, perm3, 0 /* verbose_level */);
		cout << "as permutation: " << endl;
		perm_print(cout, perm3, A->degree);
		cout << endl;

		if (!perm_is_identity(perm3, A->degree)) {
			cout << "fails the inverse test" << endl;
			exit(1);
		}
	}

	cout << "tensor_product::init test 2 passed" << endl;



	for (cnt = 0; cnt < 10; cnt++) {
		r1 = random_integer(SG->gens->len);
		r2 = random_integer(SG->gens->len);
		cout << "r1=" << r1 << endl;
		cout << "r2=" << r2 << endl;
		A->element_move(SG->gens->ith(r1), Elt1, 0);
		A->element_move(SG->gens->ith(r2), Elt2, 0);
		cout << "Elt1 = " << endl;
		A->element_print_quick(Elt1, cout);
		A->element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
		cout << "as permutation: " << endl;
		perm_print(cout, perm1, A->degree);
		cout << endl;

		cout << "Elt2 = " << endl;
		A->element_print_quick(Elt2, cout);
		A->element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
		cout << "as permutation: " << endl;
		perm_print(cout, perm2, A->degree);
		cout << endl;

		A->element_mult(Elt1, Elt2, Elt3, 0);
		cout << "Elt3 = " << endl;
		A->element_print_quick(Elt3, cout);

		A->element_invert(Elt3, Elt4, 0);
		cout << "Elt4 = Elt3^-1 = " << endl;
		A->element_print_quick(Elt4, cout);


		A->element_as_permutation(Elt3, perm3, 0 /* verbose_level */);
		cout << "as Elt3 as permutation: " << endl;
		perm_print(cout, perm3, A->degree);
		cout << endl;

		A->element_as_permutation(Elt4, perm4, 0 /* verbose_level */);
		cout << "as Elt4 as permutation: " << endl;
		perm_print(cout, perm4, A->degree);
		cout << endl;

		perm_mult(perm3, perm4, perm5, A->degree);
		cout << "perm3 * perm4= " << endl;
		perm_print(cout, perm5, A->degree);
		cout << endl;

		for (i = 0; i < A->degree; i++) {
			if (perm5[i] != i) {
				cout << "test " << cnt
						<< " failed; something is wrong" << endl;
				exit(1);
			}
		}
	}
	cout << "tensor_product::init test 3 passed" << endl;


	cout << "performing test 4:" << endl;

	int data[] = {2,0,1, 0,1,1,0, 1,0,0,1, 1,0,0,1 };
	A->make_element(Elt1, data, verbose_level);
	A->element_as_permutation(Elt1, perm1, 0 /* verbose_level */);
	cout << "as Elt1 as permutation: " << endl;
	perm_print(cout, perm1, A->degree);
	cout << endl;

	A->element_invert(Elt1, Elt2, 0);
	A->element_as_permutation(Elt2, perm2, 0 /* verbose_level */);
	cout << "as Elt2 as permutation: " << endl;
	perm_print(cout, perm2, A->degree);
	cout << endl;


	A->element_mult(Elt1, Elt2, Elt3, 0);
	cout << "Elt3 = " << endl;
	A->element_print_quick(Elt3, cout);

	perm_mult(perm1, perm2, perm3, A->degree);
	cout << "perm1 * perm2= " << endl;
	perm_print(cout, perm3, A->degree);
	cout << endl;

	for (i = 0; i < A->degree; i++) {
		if (perm3[i] != i) {
			cout << "test 4 failed; something is wrong" << endl;
			exit(1);
		}
	}

	cout << "tensor_product::init test 4 passed" << endl;

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
	FREE_int(perm1);
	FREE_int(perm2);
	FREE_int(perm3);
	FREE_int(perm4);
	FREE_int(perm5);
#endif


	Gen = NEW_OBJECT(poset_classification);

	Gen->read_arguments(argc, argv, 0);

	//Gen->prefix[0] = 0;
	sprintf(Gen->fname_base, "wreath_%d_%d_%d", nb_factors, n, q);


	Gen->depth = depth;

	VS = NEW_OBJECT(vector_space);
	VS->init(F, vector_space_dimension /* dimension */,
			verbose_level - 1);
	VS->init_rank_functions(
			wreath_rank_point_func,
			wreath_unrank_point_func,
			this,
			verbose_level - 1);


	Poset = NEW_OBJECT(poset);
	Poset->init_subspace_lattice(A0, A,
			SG,
			VS,
			verbose_level);

	if (f_v) {
		cout << "tensor_product::init before Gen->init" << endl;
		}
	Gen->init(Poset, Gen->depth /* sz */, verbose_level);
	if (f_v) {
		cout << "tensor_product::init after Gen->init" << endl;
		}


	Gen->f_print_function = TRUE;
	Gen->print_function = wreath_product_print_set;
	Gen->print_function_data = this;

	int nb_nodes = 1000;

	if (f_v) {
		cout << "tensor_product::init "
				"before Gen->init_poset_orbit_node" << endl;
		}
	Gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);
	if (f_v) {
		cout << "tensor_product::init "
				"calling Gen->init_root_node" << endl;
		}
	Gen->root[0].init_root_node(Gen, verbose_level - 1);

	//int schreier_depth;
	int f_use_invariant_subset_if_available;
	int f_debug;

	//schreier_depth = Gen->depth;
	f_use_invariant_subset_if_available = TRUE;
	f_debug = FALSE;

	int t0 = os_ticks();

	if (f_v) {
		cout << "tensor_product::init before Gen->main" << endl;
		cout << "A=";
		A->print_info();
		cout << "A0=";
		A0->print_info();
		}


	//Gen->f_allowed_to_show_group_elements = TRUE;

	Gen->main(t0,
		Gen->depth,
		f_use_invariant_subset_if_available,
		f_debug,
		verbose_level);

	set_of_sets *SoS;

	SoS = Gen->Schreier_vector_handler->get_orbits_as_set_of_sets(
			Gen->root[0].Schreier_vector, verbose_level);

	SoS->sort_all(verbose_level);
	cout << "orbits at level 1:" << endl;
	SoS->print_table();

	for (i = 0; i < SoS->nb_sets; i++) {
		cout << "Orbit " << i << " has size " << SoS->Set_size[i] << " : ";
		int_vec_print(cout, SoS->Sets[i], SoS->Set_size[i]);
		cout << endl;
		for (j = 0; j < SoS->Set_size[i]; j++) {
			a = SoS->Sets[i][j];
			cout << j << " : " << a << " : ";
			F->PG_element_unrank_modified(v, 1, vector_space_dimension, a);
			int_vec_print(cout, v, vector_space_dimension);
			cout << endl;
		}
	}

	if (f_v) {
		cout << "tensor_product::init after Gen->main" << endl;
	}
}


int wreath_rank_point_func(int *v, void *data)
{
	tensor_product *T;
	int rk;

	T = (tensor_product *) data;
	//AG_element_rank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	T->F->PG_element_rank_modified(v, 1, T->vector_space_dimension, rk);
	return rk;
}

void wreath_unrank_point_func(int *v, int rk, void *data)
{
	tensor_product *T;

	T = (tensor_product *) data;
	//AG_element_unrank(LS->Fq->q, v, 1, LS->vector_space_dimension, rk);
	T->F->PG_element_unrank_modified(v, 1, T->vector_space_dimension, rk);
}


void wreath_product_print_set(ostream &ost, int len, int *S, void *data)
{
	tensor_product *T;
	int i;

	T = (tensor_product *) data;
	cout << "set: ";
	int_vec_print(cout, S, len);
	cout << endl;
	for (i = 0; i < len; i++) {
		T->F->PG_element_unrank_modified(T->v,
				1, T->vector_space_dimension, S[i]);
		cout << S[i] << " : ";
		int_vec_print(cout, T->v, T->vector_space_dimension);
		cout << endl;
	}
}

