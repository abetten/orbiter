// ovoid_generator.C
// 
// Anton Betten
// May 16, 2011
//
//
// 
// pulled out of ovoid: Jul 30, 2018
//

#include "orbiter.h"

#include "ovoid.h"

ovoid_generator::ovoid_generator()
{
	gen = NULL;
	F = NULL;
	A = NULL;
	O = NULL;
	
	f_max_depth = FALSE;
	f_list = FALSE;
	f_poly = FALSE;
	override_poly = NULL;
	f_draw_poset = FALSE;
	f_embedded = FALSE;
	f_sideways = FALSE;

	f_read = FALSE;
	read_level = 0;

	nb_identify = 0;
	Identify_label = NULL;
	Identify_coeff = NULL;
	Identify_monomial = NULL;
	Identify_length = NULL;

	// for surface:
	//f_surface = FALSE;
}

ovoid_generator::~ovoid_generator()
{
	INT f_v = FALSE;
	INT i;

	if (f_v) {
		cout << "ovoid_generator::~ovoid_generator()" << endl;
		}
	if (A) {
		delete A;
		}
	if (F) {
		delete F;
		}

	if (Identify_label) {
		for (i = 0; i < nb_identify; i++) {
			FREE_BYTE(Identify_label[i]);
			}
		FREE_PBYTE(Identify_label);
		}
	if (Identify_coeff) {
		for (i = 0; i < nb_identify; i++) {
			FREE_INT(Identify_coeff[i]);
			}
		FREE_PINT(Identify_coeff);
		}
	if (Identify_monomial) {
		for (i = 0; i < nb_identify; i++) {
			FREE_INT(Identify_monomial[i]);
			}
		FREE_PINT(Identify_monomial);
		}
	if (Identify_length) {
		FREE_INT(Identify_length);
		}

#if 0
	if (SC) {
		delete SC;
		}
#endif
	
	if (f_v) {
		cout << "ovoid_generator::~ovoid_generator() after delete A" << endl;
		cout << "ovoid_generator::~ovoid_generator() finished" << endl;
		}
	
}

void ovoid_generator::init(int argc, const char **argv, INT &verbose_level)
{
	INT f_semilinear;
	INT f_basis = TRUE;

	F = new finite_field;
	A = new action;
	gen = new generator;

	read_arguments(argc, argv, verbose_level);


	gen->read_arguments(argc, argv, 0);

	
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 4);
	
	u = NEW_INT(d);
	v = NEW_INT(d);
	w = NEW_INT(d);
	tmp1 = NEW_INT(d);

	INT p, h;
	is_prime_power(q, p, h);


	if (h > 1) {
		f_semilinear = TRUE;
		}
	else {
		f_semilinear = FALSE;
		}


	//f_semilinear = TRUE;

	
	sprintf(prefix_with_directory, "ovoid_Q%ld_%ld_%ld", epsilon, d - 1, q);
	
	F->init_override_polynomial(q, override_poly, 0);

	INT f_siegel = TRUE;
	INT f_reflection = TRUE;
	INT f_similarity = TRUE;
	INT f_semisimilarity = TRUE;
	set_orthogonal_group_type(f_siegel, f_reflection, f_similarity, f_semisimilarity);


	cout << "ovoid_generator::init d=" << d << endl;
	cout << "ovoid_generator::init f_siegel=" << f_siegel << endl;
	cout << "ovoid_generator::init f_reflection=" << f_reflection << endl;
	cout << "ovoid_generator::init f_similarity=" << f_similarity << endl;
	cout << "ovoid_generator::init f_semisimilarity=" << f_semisimilarity << endl;
	
	A->init_orthogonal_group(epsilon, d, F, 
		TRUE /* f_on_points */, 
		FALSE /* f_on_lines */, 
		FALSE /* f_on_points_and_lines */, 
		f_semilinear, f_basis, verbose_level);
	

	action_on_orthogonal *AO;
	
	AO = A->G.AO;
	O = AO->O;

	N = O->nb_points + O->nb_lines;
	
	if (f_vv) {
		cout << "The finite field is:" << endl;
		O->F->print(TRUE);
		}

	if (f_v) {
		cout << "nb_points=" << O->nb_points << endl;
		cout << "nb_lines=" << O->nb_lines << endl;
		cout << "alpha=" << O->alpha << endl;
		}



	//A->Strong_gens->print_generators_even_odd();
	

#if 0
	if (f_surface) {

		if (n != 5) {
			cout << "surface needs n = 5" << endl;
			exit(1);
			}
		if (epsilon != 1) {
			cout << "surface needs epsilon = 1" << endl;
			exit(1);
			}

		

		if (f_v) {
			cout << "surface:" << endl;
			}

		SC = new surface_classify;

		if (f_v) {
			cout << "before SC->init" << endl;
			}
		SC->init(F, A, O, gen, verbose_level);
		if (f_v) {
			cout << "after SC->init" << endl;
			}

		}
#endif

	if (f_max_depth) {
		depth = max_depth;
		}
	else {
		if (epsilon == 1) {
			depth = i_power_j(q, m - 1) + 1;
			}
		else if (epsilon == -1) {
			depth = i_power_j(q, m + 1) + 1;
			}
		else if (epsilon == 0) {
			depth = i_power_j(q, m) + 1;
			}
		else {
			cout << "epsilon must be 0, 1, or -1" << endl;
			exit(1);
			}
		}
	

	gen->depth = depth;
	if (f_v) {
		cout << "depth = " << depth << endl;
		}
	

	if (FALSE /* f_surface*/) {
#if 0
		gen->init(A, SC->A_on_neighbors, 
			SC->stab_gens,  
			gen->depth /* sz */, 
			verbose_level - 1);

#if 1
		gen->init_check_func(callback_check_surface, 
			(void *)this /* candidate_check_data */);
#endif

		//gen->f_print_function = TRUE;
		//gen->print_function = callback_print_set;
		//gen->print_function_data = (void *) this;
	

		sprintf(gen->fname_base, "surface_%ld", q);
#endif
		}
	else {
		gen->init(A, A, 
			A->Strong_gens,  
			gen->depth /* sz */, 
			verbose_level - 1);

		gen->init_check_func(callback_check_conditions, 
			(void *)this /* candidate_check_data */);

		gen->f_print_function = TRUE;
		gen->print_function = callback_print_set;
		gen->print_function_data = (void *) this;
	

		sprintf(gen->fname_base, "ovoid_Q%ld_%ld_%ld", epsilon, n, q);
		}

	if (f_v) {
		cout << "fname_base = " << gen->fname_base << endl;
		}
	
	
	INT nb_oracle_nodes = ONE_MILLION;
	
	if (f_v) {
		cout << "calling init_oracle with " << nb_oracle_nodes << " nodes" << endl;
		}
	
	gen->init_oracle(nb_oracle_nodes, verbose_level - 1);

	if (f_v) {
		cout << "after calling init_root_node" << endl;
		}
	
	gen->root[0].init_root_node(gen, gen->verbose_level);
	if (f_v) {
		cout << "init() finished" << endl;
		}
}

void ovoid_generator::read_arguments(int argc, const char **argv, INT &verbose_level)
{
	INT i, j;
	INT f_epsilon = FALSE;
	INT f_n = FALSE;
	INT f_q = FALSE;
	
	if (argc < 1) {
		usage(argc, argv);
		exit(1);
		}
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v" << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-epsilon") == 0) {
			f_epsilon = TRUE;
			epsilon = atoi(argv[++i]);
			cout << "-epsilon " << epsilon << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			cout << "-list" << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_max_depth = TRUE;
			max_depth = atoi(argv[++i]);
			cout << "-depth " << max_depth << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			override_poly = argv[++i];
			cout << "-poly " << override_poly << endl;
			}
#if 0
		else if (strcmp(argv[i], "-surface") == 0) {
			f_surface = TRUE;
			cout << "-surface " << endl;
			}
#endif
		else if (strcmp(argv[i], "-identify") == 0) {
			if (nb_identify == 0) {
				Identify_label = NEW_PBYTE(1000);
				Identify_coeff = NEW_PINT(1000);
				Identify_monomial = NEW_PINT(1000);
				Identify_length = NEW_INT(1000);
				}
			INT coeff[1000];
			INT monomial[1000];
			INT nb_terms = 0;
			cout << "-identify " << endl;
			const BYTE *label = argv[++i];
			cout << "-identify " << label << endl;
			Identify_label[nb_identify] = NEW_BYTE(strlen(label) + 1);
			strcpy(Identify_label[nb_identify], label);
			for (j = 0; ; j++) {
				coeff[j] = atoi(argv[++i]);
				if (coeff[j] == -1) {
					break;
					}
				monomial[j] = atoi(argv[++i]);
				}
			nb_terms = j;
			Identify_coeff[nb_identify] = NEW_INT(nb_terms);
			Identify_monomial[nb_identify] = NEW_INT(nb_terms);
			Identify_length[nb_identify] = nb_terms;
			INT_vec_copy(coeff, Identify_coeff[nb_identify], nb_terms);
			INT_vec_copy(monomial, Identify_monomial[nb_identify], nb_terms);
			cout << "-identify " << Identify_label[nb_identify] << " ";
			for (j = 0; j < Identify_length[nb_identify]; j++) {
				cout << Identify_coeff[nb_identify][j] << " ";
				cout << Identify_monomial[nb_identify][j] << " ";
				}
			cout << "-1" << endl;
			nb_identify++;
			
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-read") == 0) {
			f_read = TRUE;
			read_level = atoi(argv[++i]);
			cout << "-read " << read_level << endl;
			}
		}
	if (!f_epsilon) {
		cout << "Please use option -epsilon <epsilon>" << endl;
		exit(1);
		}
	if (!f_n) {
		cout << "Please use option -n <n> to specify the projective dimension" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "Please use option -q <q>" << endl;
		exit(1);
		}
	m = Witt_index(epsilon, n);
	d = n + 1;
	cout << "epsilon=" << epsilon << endl;
	cout << "projective dimension n=" << n << endl;
	cout << "d=" << d << endl;
	cout << "q=" << q << endl;
	cout << "Witt index " << m << endl;
}

INT ovoid_generator::check_conditions(INT len, INT *S, INT verbose_level)
{
	INT f_OK = TRUE;
	INT f_collinearity_test = FALSE;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "ovoid_generator::check_conditions checking set ";
		print_set(cout, len, S);
		}
	if (!collinearity_test(S, len, verbose_level - 1)) {
		f_OK = FALSE;
		f_collinearity_test = TRUE;
		}
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "not OK because of ";
			if (f_collinearity_test) {
				cout << "collinearity test";
				}
			cout << endl;
			}
		return FALSE;
		}
}

#if 0
INT ovoid_generator::check_surface(INT len, INT *S, INT verbose_level)
{
	INT f_OK = TRUE;
	INT f_surface_test = FALSE;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "ovoid_generator::check_surface checking set ";
		print_set(cout, len, S);
		}
	if (!SC->surface_test(S, len, verbose_level - 1)) {
		f_OK = FALSE;
		f_surface_test = TRUE;
		}
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return TRUE;
		}
	else {
		if (f_v) {
			cout << "not OK because of ";
			if (f_surface_test) {
				cout << "surface test";
				}
			cout << endl;
			}
		return FALSE;
		}
}
#endif

INT ovoid_generator::collinearity_test(INT *S, INT len, INT verbose_level)
{
	INT i, x, y;
	INT f_OK = TRUE;
	INT fxy;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "collinearity test" << endl;
		}
	if (f_vv) {
		for (i = 0; i < len; i++) {
			O->unrank_point(O->v1, 1, S[i], 0);
			INT_vec_print(cout, u, n);
			cout << endl;
			}
		}
	y = S[len - 1];
	O->unrank_point(v, 1, y, 0);
	
	for (i = 0; i < len - 1; i++) {
		x = S[i];
		O->unrank_point(u, 1, x, 0);

		fxy = O->evaluate_bilinear_form(u, v, 1);
		
		if (fxy == 0) {
			f_OK = FALSE;
			if (f_vv) {
				cout << "not OK; ";
				cout << "{x,y}={" << x << "," << y << "} are collinear" << endl;
				INT_vec_print(cout, u, n);
				cout << endl;
				INT_vec_print(cout, v, n);
				cout << endl;
				cout << "fxy=" << fxy << endl;
				}
			break;
			}
		}
	
	if (f_v) {
		if (!f_OK) {
			cout << "collinearity test fails" << endl;
			}
		}
	return f_OK;
}

void ovoid_generator::print(INT *S, INT len)
{
	INT i;
	
	for (i = 0; i < len; i++) {
		for (i = 0; i < len; i++) {
			O->unrank_point(u, 1, S[i], 0);
			INT_vec_print(cout, u, n);
			cout << endl;
			}
		}
}

#if 0
void ovoid_generator::process_surfaces(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "ovoid_generator::process_surfaces" << endl;
		}
	SC->process_surfaces(nb_identify, 
		Identify_label, 
		Identify_coeff, 
		Identify_monomial, 
		Identify_length, 
		verbose_level);
	if (f_v) {
		cout << "ovoid_generator::process_surfaces done" << endl;
		}

}
#endif


