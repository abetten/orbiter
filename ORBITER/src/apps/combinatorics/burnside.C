// burnside.C
//
// Anton Betten
// February 26, 2015
//

#include "orbiter.h"


INT t0;

void do_it(INT n, INT verbose_level);
void create_matrix(matrix &M, INT i, INT *S, INT nb_classes, 
	INT *character_degree, INT *class_size, 
	INT verbose_level);
void compute_character_table(a_domain *D, INT nb_classes, INT *Omega, 
	INT *character_degree, INT *class_size, 
	INT *&character_table, INT verbose_level);
void compute_character_degrees(a_domain *D, INT goi, INT nb_classes, INT *Omega, INT *class_size, 
	INT *&character_degree, INT verbose_level);
void compute_omega(a_domain *D, INT *N0, INT nb_classes, INT *Mu, INT nb_mu, INT *&Omega, INT verbose_level);
INT compute_r0(INT *N, INT nb_classes, INT verbose_level);
void compute_multiplication_constants_center_of_group_ring(action *A, 
	action_by_conjugation *ABC, 
	schreier *Sch, INT nb_classes, INT *&N, INT verbose_level);
void compute_Distribution_table(action *A, action_by_conjugation *ABC, 
	schreier *Sch, INT nb_classes, 
	INT **Gens, INT nb_gens, INT t_max, INT *&Distribution, INT verbose_level);
void multiply_word(action *A, INT **Gens, INT *Choice, INT t, INT *Elt1, INT *Elt2, INT verbose_level);
void create_generators(action *A, INT n, INT **&Elt, INT &nb_gens, INT f_special, INT verbose_level);
void integral_eigenvalues(INT *M, INT n, 
	INT *&Lambda, 
	INT &nb_lambda, 
	INT *&Mu, 
	INT *&Mu_mult, 
	INT &nb_mu, 
	INT verbose_level);
void characteristic_poly(INT *N, INT size, unipoly &charpoly, INT verbose_level);
void double_swap(double &a, double &b);
INT double_Gauss(double *A, INT m, INT n, INT *base_cols, INT verbose_level);
void double_matrix_print(double *A, INT m, INT n);
double double_abs(double x);
void kernel_columns(INT n, INT nb_base_cols, INT *base_cols, INT *kernel_cols);
void matrix_get_kernel(double *M, INT m, INT n, INT *base_cols, INT nb_base_cols, 
	INT &kernel_m, INT &kernel_n, double *kernel);
INT double_as_INT(double x);



int main(int argc, char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_n = FALSE;
	INT n = 0;
	
	t0 = os_ticks();

	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		}
	


	if (!f_n) {
		cout << "please specify -n <n>" << endl;
		exit(1);
		}

	do_it(n, verbose_level);
	
}

void do_it(INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "do_it" << endl;
		}

	a_domain *D;

	D = new a_domain;
	D->init_integer_fractions(verbose_level);


	action *A;
	longinteger_object go;
	INT goi;
	INT *Elt;
	INT i, j;

	A = new action;
	A->init_symmetric_group(n, verbose_level);
	A->group_order(go);

	goi = go.as_INT();
	cout << "Created group Sym(" << n << ") of size " << goi << endl;

	Elt = NEW_INT(A->elt_size_in_INT);

	sims *S;

	S = A->Sims;

	for (i = 0; i < goi; i++) {
		S->element_unrank_INT(i, Elt);
		cout << "element " << i << " is ";
		A->element_print_quick(Elt, cout);
		cout << endl;
		}

	action *Aconj;

	Aconj = new action;

	cout << "Creating action by conjugation" << endl;
	
	Aconj->induced_action_by_conjugation(S, 
		S, FALSE /* f_ownership */, FALSE /* f_basis */, verbose_level);

	cout << "Creating action by conjugation done" << endl;
	
	action_by_conjugation *ABC;

	ABC = Aconj->G.ABC;

	schreier *Sch;
	strong_generators *SG;

	Sch = new schreier;

	Sch->init(Aconj);


	SG = new strong_generators;

	SG->init_from_sims(S, 0);
	
#if 0
	if (!A->f_has_strong_generators) {
		cout << "action does not have strong generators" << endl;
		exit(1);
		}
#endif

	Sch->init_generators(*SG->gens);

	cout << "Computing conjugacy classes:" << endl;
	Sch->compute_all_point_orbits(verbose_level);
	

	INT nb_classes;
	INT *class_size;

	nb_classes = Sch->nb_orbits;

	class_size = NEW_INT(nb_classes);
	
	for (i = 0; i < nb_classes; i++) {
		class_size[i] = Sch->orbit_len[i];
		}
	cout << "class sizes : ";
	INT_vec_print(cout, class_size, nb_classes);
	cout << endl;




	INT *N;
	INT r, r0;


	compute_multiplication_constants_center_of_group_ring(A, 
		ABC, 
		Sch, nb_classes, N, verbose_level);


	for (r = 0; r < nb_classes; r++) {
		cout << "N_" << r << ":" << endl;
		INT_matrix_print(N + r * nb_classes * nb_classes, nb_classes, nb_classes);
		cout << endl;
		}


	r0 = compute_r0(N, nb_classes, verbose_level);


	if (r0 == -1) {
		cout << "Did not find a matrix with the right number of distinct eigenvalues" << endl;
		exit(1);
		}


	cout << "r0=" << r0 << endl;

	INT *N0;

	N0 = N + r0 * nb_classes * nb_classes;




	INT *Lambda;
	INT nb_lambda;
	INT *Mu;
	INT *Mu_mult;
	INT nb_mu;

	cout << "N_" << r0 << ":" << endl;

	integral_eigenvalues(N0, nb_classes, 
		Lambda, 
		nb_lambda, 
		Mu, 
		Mu_mult, 
		nb_mu, 
		0 /*verbose_level*/);

	cout << "Has " << nb_mu << " distinct eigenvalues" << endl;


	cout << "We found " << nb_lambda << " integer roots, they are: " << endl;
	INT_vec_print(cout, Lambda, nb_lambda);
	cout << endl;		
	cout << "We found " << nb_mu << " distinct integer roots, they are: " << endl;
	for (i = 0; i < nb_mu; i++) {
		cout << Mu[i] << " with multiplicity " << Mu_mult[i] << endl;
		}

	INT *Omega;


	compute_omega(D, N0, nb_classes, Mu, nb_mu, Omega, verbose_level);



	cout << "Omega:" << endl;
	D->print_matrix(Omega, nb_classes, nb_classes);
	//double_matrix_print(Omega, nb_classes, nb_classes);




	INT *character_degree;


	compute_character_degrees(D, goi, nb_classes, Omega, class_size, 
		character_degree, verbose_level);


	cout << "character degrees : ";
	INT_vec_print(cout, character_degree, nb_classes);
	cout << endl;
	

	INT *character_table;


	compute_character_table(D, nb_classes, Omega, 
		character_degree, class_size, 
		character_table, verbose_level);

	cout << "character table:" << endl;
	INT_matrix_print(character_table, nb_classes, nb_classes);

	INT f_special = TRUE;
	INT **Gens;
	INT nb_gens;
	INT t_max;
	INT *Distribution;
	
	t_max = character_degree[0];
	for (i = 0; i < nb_classes; i++) {
		if (character_degree[i] > t_max) {
			t_max = character_degree[i];
			}
		}

	cout << "t_max=" << t_max << endl;

	cout << "creating generators:" << endl;
	create_generators(A, n, Gens, nb_gens, f_special, verbose_level);

	
	compute_Distribution_table(A, ABC, 
		Sch, nb_classes, 
		Gens,nb_gens, t_max, Distribution, verbose_level);


	cout << "Distribution table:" << endl;
	INT_matrix_print(Distribution + nb_classes, t_max, nb_classes);

	
	for (i = 0; i < nb_classes; i++) {
		
		cout << "character " << i << " / " << nb_classes << ":" << endl;
		INT_vec_print(cout, character_table + i * nb_classes, nb_classes);
		cout << endl;

		
		INT *S, a, t;

		S = NEW_INT(t_max + 1);
		INT_vec_zero(S, t_max + 1);

		for (t = 0; t <= t_max; t++) {
			S[t] = 0;
			for (j = 0; j < nb_classes; j++) {
				a = Distribution[t * nb_classes + j];
				if (a == 0) {
					continue;
					}
				S[t] += a * character_table[i * nb_classes + j];
				}			
			}
		cout << "S=";
		INT_vec_print(cout, S + 1, t_max);
		cout << endl;


		matrix M;

		INT /*n,*/ deg;

		//n = character_degree[i];
		
		create_matrix(M, i, S, nb_classes, 
			character_degree, class_size, 
			verbose_level);

		cout << "M=" << endl;
		cout << M << endl;

		unipoly p;


		M.determinant(p, 0 /*verbose_level*/);
		if (f_v) {
			cout << "determinant:" << p << endl;
			}

		deg = p.degree();
		if (f_v) {
			cout << "has degree " << deg << endl;
			}




		FREE_INT(S);
		}



	FREE_INT(Distribution);
	for (i = 0; i < nb_gens; i++) {
		FREE_INT(Gens[i]);
		}
	FREE_PINT(Gens);


	FREE_INT(character_table);
	FREE_INT(character_degree);
	
	FREE_INT(Omega);

	FREE_INT(Lambda);
	FREE_INT(Mu);
	FREE_INT(Mu_mult);


	FREE_INT(N);
	delete SG;
	delete Sch;
	delete Aconj;

	FREE_INT(Elt);
	delete A;
	delete D;
}

void create_matrix(matrix &M, INT i, INT *S, INT nb_classes, 
	INT *character_degree, INT *class_size, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT n, ii, j;


	if (f_v) {
		cout << "create_matrix" << endl;
		}
	n = character_degree[i];
	if (f_v) {
		cout << "n=" << n << endl;
		}
	M.m_mn_n(n + 1, n + 1);

	M.elements_to_unipoly();

	for (j = 0; j <= n; j++) {

		{
		unipoly p;

		p.x_to_the_i(j);
		M.s_ij(0, n - j) = p;
		}
		
		}
	for (ii = 1; ii <= n; ii++) {

		cout << "ii=" << ii << endl;
		
		for (j = 0; j <= ii; j++) {
			unipoly p;

			p.one();
			if (j == 0) {
				p.m_ii(0, ii);
				}
			else {
				p.m_ii(0, S[j]);
				}
			cout << "j=" << j << " p=" << p << endl;
		
			M.s_ij(ii, ii - j) = p;
			}
		}

	if (f_v) {
		cout << "create_matrix done" << endl;
		}
}

void compute_character_table(a_domain *D, INT nb_classes, INT *Omega, 
	INT *character_degree, INT *class_size, 
	INT *&character_table, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 3);
	INT i, j, w;
	
	if (f_v) {
		cout << "compute_character_table" << endl;
		}

	character_table = NEW_INT(nb_classes * nb_classes);
	
	for (i = 0; i < nb_classes; i++) {
		
		for (j = 0; j < nb_classes; j++) {

			if (f_vv) {
				cout << "i=" << i << " j=" << j << " character_degree[i]=" << character_degree[i] 
					<< " omega_ij=" << D->as_INT(D->offset(Omega, j * nb_classes + i), 0) << " class_size[j]=" << class_size[j] << endl;
				}
			
			w = character_degree[i] * D->as_INT(D->offset(Omega, j * nb_classes + i), 0);
			if (w % class_size[j]) {
				cout << "class size does not divide w" << endl;
				exit(1);
				}
			character_table[i * nb_classes + j] = w / class_size[j];
			}
		}


	if (f_v) {
		cout << "compute_character_table done" << endl;
		}
}

void compute_character_degrees(a_domain *D, INT goi, INT nb_classes, INT *Omega, INT *class_size, 
	INT *&character_degree, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, r, d, f;
	INT *A, *B, *C, *Cv, *G, *S, *Sv, *E, *F;

	if (f_v) {
		cout << "compute_character_degrees" << endl;
		}
	
	character_degree = NEW_INT(nb_classes);
	A = NEW_INT(D->size_of_instance_in_INT);
	B = NEW_INT(D->size_of_instance_in_INT);
	C = NEW_INT(D->size_of_instance_in_INT);
	Cv = NEW_INT(D->size_of_instance_in_INT);
	G = NEW_INT(D->size_of_instance_in_INT);
	S = NEW_INT(D->size_of_instance_in_INT);
	Sv = NEW_INT(D->size_of_instance_in_INT);
	E = NEW_INT(D->size_of_instance_in_INT);
	F = NEW_INT(D->size_of_instance_in_INT);

	for (i = 0; i < nb_classes; i++) {


		D->make_zero(S, 0);

		for (r = 0; r < nb_classes; r++) {
			D->copy(D->offset(Omega, r * nb_classes + i), A, 0);

			D->mult(A, A, B, 0);


			D->make_integer(C, class_size[r], 0);
			D->inverse(C, Cv, 0);


			D->mult(B, Cv, E, 0);

			D->add_apply(S, E, 0);
			}
		
		D->inverse(S, Sv, 0);
	
		D->make_integer(G, goi, 0);
		D->mult(G, Sv, F, 0);

		
		f = D->as_INT(F, 0);
		d = sqrt(f);

		if (d * d != f) {
			cout << "f is not a perfect square" << endl;
			exit(1);
			}

		if (f_vv) {
			cout << "i=" << i << " d=" << d << endl;
			}

		character_degree[i] = d;
		}
	FREE_INT(A);
	FREE_INT(B);
	FREE_INT(C);
	FREE_INT(Cv);
	FREE_INT(G);
	FREE_INT(S);
	FREE_INT(Sv);
	FREE_INT(E);
	FREE_INT(F);
	if (f_v) {
		cout << "compute_character_degrees done" << endl;
		}
}

void compute_omega(a_domain *D, INT *N0, INT nb_classes, INT *Mu, INT nb_mu, INT *&Omega, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *M;
	INT *base_cols;
	INT h, x, rk, i, j, a;

	if (f_v) {
		cout << "compute_omega" << endl;
		}
	Omega = NEW_INT(nb_classes * nb_classes * D->size_of_instance_in_INT);
	M = NEW_INT(nb_classes * nb_classes * D->size_of_instance_in_INT);
	base_cols = NEW_INT(nb_classes);
	
	for (h = 0; h < nb_mu; h++) {

		x = Mu[h];
		if (f_v) {
			cout << "eigenvalue " << h << " / " << nb_mu << " is " << x << ":" << endl;
			}
		for (i = 0; i < nb_classes; i++) {
			for (j = 0; j < nb_classes; j++) {
				a = N0[i * nb_classes + j];
				if (i == j) {
					a -= x;
					}
				D->make_integer(D->offset(M, i * nb_classes + j), a, 0);
				}
			}
		if (f_vv) {
			cout << "before get_image_and_kernel:" << endl;
			D->print_matrix(M, nb_classes, nb_classes);
			//double_matrix_print(M, nb_classes, nb_classes);
			}

		D->get_image_and_kernel(M, nb_classes, rk, verbose_level);
		
		//rk = double_Gauss(M, nb_classes, nb_classes, base_cols, 0 /*verbose_level */);

		if (f_vv) {
			cout << "after get_image_and_kernel:" << endl;
			//double_matrix_print(M, nb_classes, nb_classes);
			D->print_matrix(M, nb_classes, nb_classes);

			cout << "after get_image_and_kernel, rk=" << rk << endl;
			}

		if (rk != nb_classes - 1) {
			cout << "rk != nb_classes - 1" << endl;
			exit(1);
			}

		INT *b, *c;

		b = NEW_INT(D->size_of_instance_in_INT);
		c = NEW_INT(D->size_of_instance_in_INT);
		D->copy(D->offset(M, (nb_classes - 1) * nb_classes), b, 0);
		D->inverse(b, c, 0);

		cout << "c=";
		D->print(c);
		cout << endl;
		
		for (i = 0; i < nb_classes; i++) {
			D->mult_apply(D->offset(M, (nb_classes - 1) * nb_classes + i), c, 0);
			}

		if (f_vv) {
			cout << "after rescaling:" << endl;
			D->print_matrix(M, nb_classes, nb_classes);
			}
		for (i = 0; i < nb_classes; i++) {
			D->copy(D->offset(M, (nb_classes - 1) * nb_classes + i), D->offset(Omega, i * nb_classes + h), 0);
			}
		FREE_INT(b);
		FREE_INT(c);

#if 0
		INT kernel_m, kernel_n;
		double a, b;
		double *kernel;
		
		kernel = new double[nb_classes * nb_classes];
		matrix_get_kernel(M, nb_classes, nb_classes, base_cols, rk, 
			kernel_m, kernel_n, kernel);

		if (f_vv) {
			cout << "the kernel is:" << endl;
			double_matrix_print(kernel, nb_classes, 1);
			}

		a = kernel[0];
		if (double_abs(a) < 0.000001) {
			cout << "The first entry of the eigenvector is zero" << endl;
			exit(1);
			}
		b = 1. / a;
		for (i = 0; i < nb_classes; i++) {
			kernel[i] *= b;
			}

		for (i = 0; i < nb_classes; i++) {
			Omega[i * nb_classes + h] = kernel[i];
			}

		delete [] kernel;
#endif


		}


	if (f_vv) {
		cout << "Omega:" << endl;
		D->print_matrix(Omega, nb_classes, nb_classes);
		}

	FREE_INT(M);
	FREE_INT(base_cols);
	if (f_v) {
		cout << "compute_omega done" << endl;
		}
}

INT compute_r0(INT *N, INT nb_classes, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 3);
	INT r, r0, i;
	
	if (f_v) {
		cout << "compute_r0" << endl;
		}
	r0 = -1;

	for (r = 0; r < nb_classes; r++) {

		INT *Lambda;
		INT nb_lambda;
		INT *Mu;
		INT *Mu_mult;
		INT nb_mu;

		if (f_vv) {
			cout << "N_" << r << ":" << endl;
			}

		integral_eigenvalues(N + r * nb_classes * nb_classes, nb_classes, 
			Lambda, 
			nb_lambda, 
			Mu, 
			Mu_mult, 
			nb_mu, 
			0 /*verbose_level*/);


		if (f_vv) {
			cout << "Has " << nb_mu << " distinct eigenvalues" << endl;


			cout << "We found " << nb_lambda << " integer roots, they are: " << endl;
			INT_vec_print(cout, Lambda, nb_lambda);
			cout << endl;		
			cout << "We found " << nb_mu << " distinct integer roots, they are: " << endl;
			for (i = 0; i < nb_mu; i++) {
				cout << Mu[i] << " with multiplicity " << Mu_mult[i] << endl;
				}
			}

		if (nb_mu == nb_classes) {
			r0 = r;
			}


		FREE_INT(Lambda);
		FREE_INT(Mu);
		FREE_INT(Mu_mult);


		}
	if (f_v) {
		cout << "compute_r0 done" << endl;
		}
	return r0;
}

void compute_multiplication_constants_center_of_group_ring(action *A, 
	action_by_conjugation *ABC, 
	schreier *Sch, INT nb_classes, INT *&N, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT r, rl, rf, s, sl, sf, i, a, j, b, c, idx, t, tf; //, tl;


	if (f_v) {
		cout << "compute_multiplication_constants_center_of_group_ring" << endl;
		}

	N = NEW_INT(nb_classes * nb_classes * nb_classes);
	INT_vec_zero(N, nb_classes * nb_classes * nb_classes);
	
	
	for (r = 0; r < nb_classes; r++) {
		rl = Sch->orbit_len[r];
		rf = Sch->orbit_first[r];

		for (s = 0; s < nb_classes; s++) {
			sl = Sch->orbit_len[s];
			sf = Sch->orbit_first[s];

			
			for (i = 0; i < rl; i++) {
				a = Sch->orbit[rf + i];

				for (j = 0; j < sl; j++) {
					b = Sch->orbit[sf + j];
				
					c = ABC->multiply(A, a, b, 0 /*verbose_level*/);


					idx = Sch->orbit_inv[c];
					
					t = Sch->orbit_number(c); //Sch->orbit_no[idx];
					
					tf = Sch->orbit_first[t];
					//tl = Sch->orbit_len[t];

					if (idx == tf) {
						N[r * nb_classes * nb_classes + s * nb_classes + t]++;
						}
					}
				}
			}
		}
	if (f_v) {
		cout << "compute_multiplication_constants_center_of_group_ring done" << endl;
		}
}

void compute_Distribution_table(action *A, action_by_conjugation *ABC, 
	schreier *Sch, INT nb_classes, 
	INT **Gens, INT nb_gens, INT t_max, INT *&Distribution, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT *Elt1;
	INT *Elt2;
	INT *Choice;
	INT *Nb;
	INT t, h, i, /*idx,*/ j;

	if (f_v) {
		cout << "compute_Distribution_table" << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	
	Choice = NEW_INT(t_max);
	Distribution = NEW_INT((t_max + 1) * nb_classes);
	INT_vec_zero(Distribution, (t_max + 1) * nb_classes);
	Nb = NEW_INT(t_max + 1);

	for (t = 1; t <= t_max; t++) {
		Nb[t] = i_power_j(nb_gens, t);
		}

	if (f_v) {
		cout << "Nb : ";
		INT_vec_print(cout, Nb + 1, t_max);
		cout << endl;
		}
	
	for (t = 1; t <= t_max; t++) {
		cout << "t=" << t << " Nb[t]=" << Nb[t] << endl;
		for (h = 0; h < Nb[t]; h++) {
			AG_element_unrank(nb_gens, Choice, 1, t, h);

			if (f_vvv) {
				cout << "h=" << h << " Choice=";
				INT_vec_print(cout, Choice, t);
				cout << endl;
				}

			multiply_word(A, Gens, Choice, t, Elt1, Elt2, verbose_level);

			i = ABC->rank(Elt1);


			//idx = Sch->orbit_inv[i];
					
			j = Sch->orbit_number(i); // Sch->orbit_no[idx];
					
			
			if (f_vvv) {
				cout << "word:";
				A->element_print(Elt1, cout);
				cout << " has rank " << i << " and belongs to class " << j;
				cout << endl;
				}

			Distribution[t * nb_classes + j]++;
			}

		if (f_v) {
			cout << "after t=" << t << " Distribution:" << endl;
			INT_matrix_print(Distribution, t + 1, nb_classes);
			}
		}

	FREE_INT(Choice);
	FREE_INT(Nb);
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	
	if (f_v) {
		cout << "compute_Distribution_table done" << endl;
		}
}


void multiply_word(action *A, INT **Gens, INT *Choice, INT t, INT *Elt1, INT *Elt2, INT verbose_level)
{
	INT i;
	
	A->element_move(Gens[Choice[0]], Elt1, 0);
	for (i = 1; i < t; i++) {
		A->element_mult(Elt1, Gens[Choice[i]], Elt2, 0);
		A->element_move(Elt2, Elt1, 0);
		}
}

void create_generators(action *A, INT n, INT **&Elt, INT &nb_gens, INT f_special, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	INT *v;
	
	if (f_v) {
		cout << "create_generators" << endl;
		}
	nb_gens = n - 1;
	Elt = NEW_PINT(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		Elt[i] = NEW_INT(A->elt_size_in_INT);
		}
	v = NEW_INT(n);


	if (f_special) {
		for (i = 0; i < nb_gens; i++) {
			for (j = 0; j < n; j++) {
				v[j] = j;
				}
			v[0] = i + 1;
			v[i + 1] = 0;
			A->make_element(Elt[i], v, 0 /* verbose_level */);
			}
		}
	else {
		for (i = 0; i < nb_gens; i++) {
			for (j = 0; j < n; j++) {
				v[j] = j;
				}
			v[i] = i + 1;
			v[i + 1] = i;
			A->make_element(Elt[i], v, 0 /* verbose_level */);
			}
		}
	cout << "generators:" << endl;
	for (i = 0; i < nb_gens; i++) {
		cout << "generator " << i << ":" << endl;
		A->element_print(Elt[i], cout);
		cout << endl;
		}

	FREE_INT(v);

}


void integral_eigenvalues(INT *M, INT n, 
	INT *&Lambda, 
	INT &nb_lambda, 
	INT *&Mu, 
	INT *&Mu_mult, 
	INT &nb_mu, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	unipoly charpoly;
	INT *A;
	INT *B;
	INT i, deg, a, x;

	if (f_v) {
		cout << "integral_eigenvalues" << endl;
		}
	characteristic_poly(M, n, charpoly, 0 /*verbose_level*/);
	if (f_v) {
		cout << "characteristic polynomial:" << charpoly << endl;
		}

	deg = charpoly.degree();
	if (f_v) {
		cout << "has degree " << deg << endl;
		}

	A = NEW_INT(deg + 1);
	B = NEW_INT(deg + 1);
		
	for (i = 0; i <= deg; i++) {
		A[i] = charpoly.s_ii(i);
		}
	if (f_v) {
		cout << "coeffs : ";
		INT_vec_print(cout, A, deg + 1);
		cout << endl;
		}


	Lambda = NEW_INT(deg);
	Mu = NEW_INT(deg);
	Mu_mult = NEW_INT(deg);
	nb_lambda = 0;
	nb_mu = 0;

	for (x = -100; x < 100; x++) {
		a = A[deg];
		for (i = deg - 1; i >= 0; i--) {
			a *= x;
			a += A[i];
			}	
		if (a == 0) {
			if (f_v) {
				cout << "Found integer root " << x << endl;
				}
			Lambda[nb_lambda++] = x;
			if (nb_mu && Mu[nb_mu - 1] == x) {
				if (f_v) {
					cout << "The root is a multiple root" << endl;
					}
				Mu_mult[nb_mu - 1]++;
				}
			else {
				Mu[nb_mu] = x;
				Mu_mult[nb_mu] = 1;
				nb_mu++;
				}
			
			for (i = deg - 1; i >= 0; i--) {
				B[i] = A[i + 1];
				A[i] = A[i] + x * B[i];
				if (i == 0 && A[0]) {
					cout << "division unsuccessful" << endl;
					exit(1);
					}
				}
			INT_vec_copy(B, A, deg);
			deg--;
			if (f_v) {
				cout << "after dividing off, the polynomial is: ";
				INT_vec_print(cout, A, deg + 1);
				cout << endl;
				}

			x--; // try x again
			}
		}


	if (f_v) {
		cout << "after dividing off integer roots, the polynomial is: ";
		INT_vec_print(cout, A, deg + 1);
		cout << endl;
		}

	if (f_v) {
		cout << "We found " << nb_lambda << " integer roots, they are: " << endl;
		INT_vec_print(cout, Lambda, nb_lambda);
		cout << endl;		
		cout << "We found " << nb_mu << " distinct integer roots, they are: " << endl;
		for (i = 0; i < nb_mu; i++) {
			cout << Mu[i] << " with multiplicity " << Mu_mult[i] << endl;
			}
		}

	FREE_INT(A);
	FREE_INT(B);
}

void characteristic_poly(INT *N, INT size, unipoly &charpoly, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, k, a;
	matrix M, M1, P, Pv, Q, Qv, S, T;
	
	if (f_v) {
		cout << "characteristic_poly" << endl;
		}
	M.m_mn(size, size);
	k = 0;
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a = N[k++];
			M.m_iji(i, j, a);
			}
		}
	if (f_vv) {
		cout << "M=" << endl;
		cout << M << endl;
		}


	M.elements_to_unipoly();
	M.minus_X_times_id();


#if 0
	M1 = M;
	cout << "M - x * Id=" << endl << M << endl;
	M.smith_normal_form(P, Pv, Q, Qv, verbose_level);

	cout << "the Smith normal form is:" << endl;
	cout << M << endl;

	S.mult(P, Pv);
	cout << "P * Pv=" << endl << S << endl;

	S.mult(Q, Qv);
	cout << "Q * Qv=" << endl << S << endl;

	S.mult(P, M1);
	cout << "T.mult(S, Q):" << endl;
	T.mult(S, Q);
	cout << "T=" << endl << T << endl;
#endif

	//INT deg;
	//INT l, lv, b;


	M.determinant(charpoly, verbose_level);
	//charpoly = M.s_ij(size - 1, size - 1);
	
	if (f_v) {
		cout << "characteristic polynomial:" << charpoly << endl;
		}
	//deg = charpoly.degree();
	//cout << "has degree " << deg << endl;

#if 0
	for (i = 0; i <= deg; i++) {
		b = charpoly.s_ii(i);
		if (b > q2) {
			b -= q;
			}
		//c = Fq.mult(b, lv);
		charpoly.m_ii(i, b);
		}

	cout << "characteristic polynomial:" << charpoly << endl;
#endif

	if (f_v) {
		cout << "characteristic_poly done" << endl;
		}
}


void double_swap(double &a, double &b)
{
	double c;

	c = a;
	a = b;
	b = c;
}

INT double_Gauss(double *A, INT m, INT n, INT *base_cols, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	double pivot, pivot_inv, z, f, a, b, c, p;
	INT i, j, k, jj, rank, idx;
	
	if (f_v) {
		cout << "double_Gauss" << endl;
		}
	i = 0;
	for (j = 0; j < n; j++) {
		if (f_vv) {
			cout << "j=" << j << endl;
			double_matrix_print(A, m, n);
			}
		// search for pivot element: 
		idx = -1;
		for (k = i; k < m; k++) {
			if (idx == -1) {
				p = A[k * n + j];
				idx = k;
				}
			else {
				if (double_abs(A[k * n + j]) > double_abs(p)) {
					p = A[k * n + j];
					idx = k;
					}
				}
			} // next k
		if (f_v) {
			cout << "column " << i << " pivot is " << p << " in row " << idx << endl;
			}
		
		if (idx == -1 || double_abs(p) < 0.00001) { // no pivot found 
			if (f_v) {
				cout << "no pivot found" << endl;
				}
			continue; // increase j, leave i constant
			}
		else {
			k = idx;
			// pivot element found: 
			if (k != i) {
				for (jj = j; jj < n; jj++) {
					double_swap(A[i * n + jj], A[k * n + jj]);
					}
				}
			}
		
		if (f_vv) {
			cout << "row " << i << " pivot in row " << k << " colum " << j << endl;
			double_matrix_print(A, m, n);
			}
		
		base_cols[i] = j;
		//if (FALSE) {
		//	cout << "."; cout.flush();
		//	}

		pivot = A[i * n + j];
		if (f_vv) {
			cout << "pivot=" << pivot << endl;
			}
		//pivot_inv = inv_table[pivot];
		pivot_inv = 1. / pivot;
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv << endl;
			}
		// make pivot to 1: 
		for (jj = j; jj < n; jj++) {
			A[i * n + jj] *= pivot_inv;
			}
		if (f_vv) {
			cout << "pivot=" << pivot << " pivot_inv=" << pivot_inv 
				<< " made to one: " << A[i * n + j] << endl;
			double_matrix_print(A, m, n);
			}

		
		// do the gaussian elimination: 

		if (f_vv) {
			cout << "doing elimination in column " << j << " from row " << i + 1 << " to row " << m - 1 << ":" << endl;
			}
		for (k = i + 1; k < m; k++) {
			if (f_vv) {
				cout << "k=" << k << endl;
				}
			z = A[k * n + j];
			if (double_abs(z) < 0.0000000001) {
				continue;
				}
			f = z;
			//A[k * n + j] = 0;
			if (f_vv) {
				cout << "eliminating row " << k << endl;
				}
			for (jj = j; jj < n; jj++) {
				a = A[i * n + jj];
				b = A[k * n + jj];
				// c := b + f * a
				//    = b - z * a              if !f_special 
				//      b - z * pivot_inv * a  if f_special 
				c = b - f * a;
				A[k * n + jj] = c;
				}
			if (f_vv) {
				double_matrix_print(A, m, n);
				}
			}
		i++;
		} // next j 
	rank = i;


	for (i = rank - 1; i >= 0; i--) {
		if (f_v) {
			cout << "."; cout.flush();
			}
		j = base_cols[i];
		a = A[i * n + j];

		// do the gaussian elimination in the upper part: 
		for (k = i - 1; k >= 0; k--) {
			z = A[k * n + j];
			if (z == 0) {
				continue;
				}
			//A[k * n + j] = 0;
			for (jj = j; jj < n; jj++) {
				a = A[i * n + jj];
				b = A[k * n + jj];
				c = - z * a;
				c += b;
				A[k * n + jj] = c;
				}
			} // next k
		} // next i

	return rank;
}

void double_matrix_print(double *A, INT m, INT n)
{
	INT i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			cout << setw(10) << A[i * n + j] << " ";
			}
		cout << endl;
		}
}

double double_abs(double x)
{
	if (x < 0) {
		return - x;
		}
	else {
		return x;
		}
}

void kernel_columns(INT n, INT nb_base_cols, INT *base_cols, INT *kernel_cols)
{
	INT i, j, k;
	
	j = k = 0;
	for (i = 0; i < n; i++) {
		if (j < nb_base_cols && i == base_cols[j]) {
			j++;
			continue;
			}
		kernel_cols[k++] = i;
		}
}

void matrix_get_kernel(double *M, INT m, INT n, INT *base_cols, INT nb_base_cols, 
	INT &kernel_m, INT &kernel_n, double *kernel)
	// kernel must point to the appropriate amount of memory! (at least n * (n - nb_base_cols) INT's)
{
	INT r, k, i, j, ii, iii, a, b;
	INT *kcol;
	
	r = nb_base_cols;
	k = n - r;
	kernel_m = n;
	kernel_n = k;
	
	kcol = NEW_INT(k);
	
	ii = 0;
	j = 0;
	if (j < r) {
		b = base_cols[j];
		}
	else {
		b = -1;
		}
	for (i = 0; i < n; i++) {
		if (i == b) {
			j++;
			if (j < r) {
				b = base_cols[j];
				}
			else {
				b = -1;
				}
			}
		else {
			kcol[ii] = i;
			ii++;
			}
		}
	if (ii != k) {
		cout << "matrix_get_kernel ii != k" << endl;
		exit(1);
		}
	//cout << "kcol = " << kcol << endl;
	ii = 0;
	j = 0;
	if (j < r) {
		b = base_cols[j];
		}
	else {
		b = -1;
		}
	for (i = 0; i < n; i++) {
		if (i == b) {
			for (iii = 0; iii < k; iii++) {
				a = kcol[iii];
				kernel[i * kernel_n + iii] = M[j * n + a];
				}
			j++;
			if (j < r) {
				b = base_cols[j];
				}
			else {
				b = -1;
				}
			}
		else {
			for (iii = 0; iii < k; iii++) {
				if (iii == ii) {
					kernel[i * kernel_n + iii] = -1.;
					}
				else {
					kernel[i * kernel_n + iii] = 0;
					}
				}
			ii++;
			}
		}
	FREE_INT(kcol);
}


INT double_as_INT(double x)
{
	INT a;
	double a1, a2;

	a = (INT) (x);
	a1 = (double)a - 0.000001;
	a2 = (double)a + 0.000001;
	if (a1 < a && a < a2) {
		return a;
		}
	cout << "error in double_as_INT" << endl;
	exit(1);
}


