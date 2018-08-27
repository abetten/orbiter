// group_ring.C
//
// Anton Betten
// March 8, 2015
//

#include "orbiter.h"

void do_it_for_sym_n(INT n, INT verbose_level);
void do_it_for_sym_4(INT n, INT verbose_level);

INT t0;

int main(int argc, char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_n = FALSE;
	INT n = 0;
	//INT f_part = FALSE;
	//INT parts[1000];
	//INT nb_parts = 0;
	//INT s;
	
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
#if 0
		else if (strcmp(argv[i], "-part") == 0) {
			f_part = TRUE;
			while (TRUE) {
				parts[nb_parts] = atoi(argv[++i]);
				if (parts[nb_parts] == -1) {
					break;
					}
				nb_parts++;
				}
			cout << "-part ";
			INT_vec_print(cout, parts, nb_parts);
			cout << endl;
			}
#endif

		}
	


	if (!f_n) {
		cout << "please specify -n <n>" << endl;
		exit(1);
		}

	if (n == 4) {
		do_it_for_sym_4(n, verbose_level);
		}
	else {
		do_it_for_sym_n(n, verbose_level);
		}


#if 0
	if (!f_part) {
		cout << "please specify -part <a1> ... <al> -1" << endl;
		exit(1);
		}


	s = 0;
	for (i = 0; i < nb_parts; i++) {
		s += parts[i];
		}
	if (s != n) {
		cout << "The partition does not add up to n" << endl;
		exit(1);
		}
	for (i = 1; i < nb_parts; i++) {
		if (parts[i] > parts[i - 1]) {
			cout << "the entries in the partition must be decreasing" << endl;
			exit(1);
			}
		}
#endif


#if 0
	a_domain D;
	INT *elt1, *elt2, *elt3;

	D.init_integer_fractions(verbose_level);

	elt1 = NEW_INT(D.size_of_instance_in_INT);
	elt2 = NEW_INT(D.size_of_instance_in_INT);
	elt3 = NEW_INT(D.size_of_instance_in_INT);

	elt1[0] = 2;
	elt1[1] = 3;
	elt2[0] = 3;
	elt2[1] = 5;
	D.add(elt1, elt2, elt3, 0);
	
	D.print(elt1);
	cout << " + ";
	D.print(elt2);
	cout << " = ";
	D.print(elt3);
	cout << endl;

	D.power(elt1, elt2, 4, 0);
	D.print(elt1);
	cout << " ^ " << 4 << " = ";
	D.print(elt2);
	cout << endl;

	FREE_INT(elt1);
	FREE_INT(elt2);
	FREE_INT(elt3);
#endif


	}


void do_it_for_sym_n(INT n, INT verbose_level)
{
	young *Y;

	Y = NEW_OBJECT(young);

	Y->init(n, verbose_level);



	INT *elt1, *elt2, *h_alpha, *elt4, *elt5, *elt6, *elt7;
	
	group_ring_element_create(Y->A, Y->S, elt1);
	group_ring_element_create(Y->A, Y->S, elt2);
	group_ring_element_create(Y->A, Y->S, h_alpha);
	group_ring_element_create(Y->A, Y->S, elt4);
	group_ring_element_create(Y->A, Y->S, elt5);
	group_ring_element_create(Y->A, Y->S, elt6);
	group_ring_element_create(Y->A, Y->S, elt7);



	INT *part;
	INT *parts;

	INT *Base;
	INT *Base_inv;
	INT *Fst;
	INT *Len;
	INT cnt, s, i, j;

	part = NEW_INT(n);
	parts = NEW_INT(n);
	Fst = NEW_INT(Y->goi);
	Len = NEW_INT(Y->goi);
	Base = NEW_INT(Y->goi * Y->goi * Y->D->size_of_instance_in_INT);
	Base_inv = NEW_INT(Y->goi * Y->goi * Y->D->size_of_instance_in_INT);
	s = 0;
	Fst[0] = 0;
	
		// create the first partition in exponential notation:
	partition_first(part, n);
	cnt = 0;


	while (TRUE) {
		INT nb_parts;

		// turn the partition from exponential notation into the list of parts:
		// the large parts come first.
		nb_parts = 0;
		for (i = n - 1; i >= 0; i--) {
			for (j = 0; j < part[i]; j++) {
				parts[nb_parts++] = i + 1;
				}
			}

		cout << "partition ";
		INT_vec_print(cout, parts, nb_parts);
		cout << endl;


			// Create the young symmetrizer based on the partition.
			// We do the very first tableau for this partition.

		INT *tableau;

		tableau = NEW_INT(n);
		for (i = 0; i < n; i++) {
			tableau[i] = i;
			}
		Y->young_symmetrizer(parts, nb_parts, tableau, elt1, elt2, h_alpha, verbose_level);
		FREE_INT(tableau);

		
		cout << "h_alpha =" << endl;
		group_ring_element_print(Y->A, Y->S, h_alpha);
		cout << endl;


		group_ring_element_copy(Y->A, Y->S, h_alpha, elt4);
		group_ring_element_mult(Y->A, Y->S, elt4, elt4, elt5);

		cout << "h_alpha * h_alpha=" << endl;
		group_ring_element_print(Y->A, Y->S, elt5);
		cout << endl;

		INT *Module_Base;
		INT *base_cols;
		INT rk;

	
		Y->create_module(h_alpha, 
			Module_Base, base_cols, rk, 
			verbose_level);
		
		cout << "Module_Basis=" << endl;
		Y->D->print_matrix(Module_Base, rk, Y->goi);


		for (i = 0; i < rk; i++) {
			for (j = 0; j < Y->goi; j++) {
				Y->D->copy(Y->D->offset(Module_Base, i * Y->goi + j), Y->D->offset(Base, s * Y->goi + j), 0);
				}
			s++;
			}
		Len[cnt] = s - Fst[cnt];
		Fst[cnt + 1] = s;

		Y->create_representations(Module_Base, base_cols, rk, verbose_level);


		FREE_INT(Module_Base);
		FREE_INT(base_cols);
		

			// create the next partition in exponential notation:
		if (!partition_next(part, n)) {
			break;
			}
		cnt++;
		}

	cout << "Basis of submodule=" << endl;
	Y->D->print_matrix(Base, s, Y->goi);


	FREE_INT(part);
	FREE_INT(parts);
	FREE_INT(Fst);
	FREE_INT(Len);
	cout << "before freeing Base" << endl;
	FREE_INT(Base);
	FREE_INT(Base_inv);
	cout << "before freeing Y" << endl;
	FREE_OBJECT(Y);
	cout << "before freeing elt1" << endl;
	FREE_INT(elt1);
	FREE_INT(elt2);
	FREE_INT(h_alpha);
	FREE_INT(elt4);
	FREE_INT(elt5);
	FREE_INT(elt6);
	FREE_INT(elt7);
	return;
}

void do_it_for_sym_4(INT n, INT verbose_level)
{
	young *Y;

	Y = NEW_OBJECT(young);

	Y->init(n, verbose_level);



	INT *elt1, *elt2, *h_alpha, *elt4, *elt5, *elt6, *elt7;
	
	group_ring_element_create(Y->A, Y->S, elt1);
	group_ring_element_create(Y->A, Y->S, elt2);
	group_ring_element_create(Y->A, Y->S, h_alpha);
	group_ring_element_create(Y->A, Y->S, elt4);
	group_ring_element_create(Y->A, Y->S, elt5);
	group_ring_element_create(Y->A, Y->S, elt6);
	group_ring_element_create(Y->A, Y->S, elt7);



	INT *part;
	INT *parts;

	INT *Base;
	INT *Base_inv;
	INT *Fst;
	INT *Len;
	INT cnt, s, i, j;

	part = NEW_INT(n);
	parts = NEW_INT(n);
	Fst = NEW_INT(Y->goi);
	Len = NEW_INT(Y->goi);
	Base = NEW_INT(Y->goi * Y->goi * Y->D->size_of_instance_in_INT);
	Base_inv = NEW_INT(Y->goi * Y->goi * Y->D->size_of_instance_in_INT);
	s = 0;
	Fst[0] = 0;
	
		// create the first partition in exponential notation:
	//partition_first(part, n);
	cnt = 0;

	INT Part[10][5] = {
		{4, -1, 0, 0, 0},
		{3, 1, -1, 0, 0},
		{3, 1, -1, 0, 0},
		{3, 1, -1, 0, 0},
		{2, 2, -1, 0, 0},
		{2, 2, -1, 0, 0},
		{2, 1, 1, -1, 0},
		{2, 1, 1, -1, 0},
		{2, 1, 1, -1, 0},
		{1, 1, 1, 1, -1},
			};
	INT Tableau[10][4] = {
		{0,1,2,3},
		{0,1,2,3}, {0,1,3,2}, {0,2,3,1}, 
		{0,1,2,3}, {0,2,1,3},
		{0,1,2,3}, {0,2,1,3}, {0,3,1,2}, 
		{0,1,2,3}
		};

	for(cnt = 0; cnt < 10; cnt++) {
		INT nb_parts;

		// turn the partition from exponential notation into the list of parts:
		// the large parts come first.
		nb_parts = 0;
		for (i = 0; i < 4; i++) {
			parts[nb_parts] = Part[cnt][i];
			if (parts[nb_parts] == -1) {
				break;
				}
			nb_parts++;
			}

		cout << "partition ";
		INT_vec_print(cout, parts, nb_parts);
		cout << endl;


			// Create the young symmetrizer based on the partition.
			// We do the very first tableau for this partition.

		Y->young_symmetrizer(parts, nb_parts, Tableau[cnt], elt1, elt2, h_alpha, verbose_level);

		
		cout << "h_alpha =" << endl;
		group_ring_element_print(Y->A, Y->S, h_alpha);
		cout << endl;


		group_ring_element_copy(Y->A, Y->S, h_alpha, elt4);
		group_ring_element_mult(Y->A, Y->S, elt4, elt4, elt5);

		cout << "h_alpha * h_alpha=" << endl;
		group_ring_element_print(Y->A, Y->S, elt5);
		cout << endl;

		INT *Module_Base;
		INT *base_cols;
		INT rk;

	
		Y->create_module(h_alpha, 
			Module_Base, base_cols, rk, 
			verbose_level);
		
		cout << "Module_Basis=" << endl;
		Y->D->print_matrix(Module_Base, rk, Y->goi);


		for (i = 0; i < rk; i++) {
			for (j = 0; j < Y->goi; j++) {
				Y->D->copy(Y->D->offset(Module_Base, i * Y->goi + j), Y->D->offset(Base, s * Y->goi + j), 0);
				}
			s++;
			}
		Len[cnt] = s - Fst[cnt];
		Fst[cnt + 1] = s;

		Y->create_representations(Module_Base, base_cols, rk, verbose_level);


		FREE_INT(Module_Base);
		FREE_INT(base_cols);
		

		}

	cout << "Basis of submodule=" << endl;
	//Y->D->print_matrix(Base, s, Y->goi);
	Y->D->print_matrix_for_maple(Base, s, Y->goi);

	FREE_INT(part);
	FREE_INT(parts);
	FREE_INT(Fst);
	FREE_INT(Len);
	cout << "before freeing Base" << endl;
	FREE_INT(Base);
	FREE_INT(Base_inv);
	cout << "before freeing Y" << endl;
	FREE_OBJECT(Y);
	cout << "before freeing elt1" << endl;
	FREE_INT(elt1);
	FREE_INT(elt2);
	FREE_INT(h_alpha);
	FREE_INT(elt4);
	FREE_INT(elt5);
	FREE_INT(elt6);
	FREE_INT(elt7);
	return;
}



#if 0
	INT h;

	INT *Rep;
	INT *Rep2;
	INT *Rep3;
	INT sz;
	INT *Mu;


	Y->D->complete_basis(Base, s, Y->goi, 0 /*verbose_level*/);

	cout << "after image_and_complement" << endl;
	cout << "Base=" << endl;
	Y->D->print_matrix(Base, Y->goi, Y->goi);

	Y->D->matrix_inverse(Base, Base_inv, Y->goi, 0 /* verbose_level */);

	cout << "inverse basis of submodule=" << endl;
	Y->D->print_matrix(Base_inv, Y->goi, Y->goi);





	sz = Y->goi * Y->goi * Y->D->size_of_instance_in_INT;
	Rep = NEW_INT(Y->goi * sz);
	Rep2 = NEW_INT(Y->goi * sz);
	Rep3 = NEW_INT(Y->goi * sz);

	for (h = 0; h < Y->goi; h++) {
		
		Y->create_representation(Base, Base_inv, Y->goi, h, Rep + h * sz, 0 /*verbose_level*/);

		cout << "Group element " << h << " is represented by" << endl;
		Y->D->print_matrix(Rep + h * sz, Y->goi, Y->goi);

		}

	INT N, k, r;
	INT *minus_Mu;
	INT *Zero, *I1, *I2;
	INT *T, *Tv;
	
	N = Y->goi;
	k = s;
	r = N - k;

	Y->Maschke(Rep, 
		Y->goi /* dim_of_module */, s /* dim_of_submodule */, 
		Mu, 
		verbose_level);

	
	minus_Mu = NEW_INT(r * k * Y->D->size_of_instance_in_INT);
	I1 = NEW_INT(k * k * Y->D->size_of_instance_in_INT);
	I2 = NEW_INT(r * r * Y->D->size_of_instance_in_INT);
	Zero = NEW_INT(k * r * Y->D->size_of_instance_in_INT);
	for (i = 0; i < r * k; i++) {
		Y->D->copy(Y->D->offset(Mu, i), Y->D->offset(minus_Mu, i), 0);
		}
	Y->D->negate_vector(minus_Mu, r * k, 0);
	Y->D->make_zero_vector(I1, k * k, 0);
	Y->D->make_zero_vector(I2, r * r, 0);
	Y->D->make_zero_vector(Zero, k * r, 0);
	for (i = 0; i < k; i++) {
		Y->D->make_one(Y->D->offset(I1, i * k + i), 0);
		}
	for (i = 0; i < r; i++) {
		Y->D->make_one(Y->D->offset(I2, i * r + i), 0);
		}
	
	T = NEW_INT(N * N * Y->D->size_of_instance_in_INT);
	Tv = NEW_INT(N * N * Y->D->size_of_instance_in_INT);
	Y->D->make_block_matrix_2x2(T, N, k, I1, Zero, Mu, I2, verbose_level);
	Y->D->make_block_matrix_2x2(Tv, N, k, I1, Zero, minus_Mu, I2, verbose_level);

	cout << "T=" << endl;
	Y->D->print_matrix(T, N, N);
	cout << "Tv=" << endl;
	Y->D->print_matrix(Tv, N, N);


	for (h = 0; h < Y->goi; h++) {

		Y->D->mult_matrix3(Tv, Rep + h * sz, T, Rep2 + h * sz, N, 0);

		}

	cout << "The transformed representation is:" << endl;
	for (h = 0; h < Y->goi; h++) {
		cout << "Representation of element " << h << ":" << endl;
		Y->D->print_matrix(Rep2 + h * sz, N, N);
		}



	cout << "Base=" << endl;
	Y->D->print_matrix(Base, Y->goi, Y->goi);

	Y->D->matrix_inverse(Base, Base_inv, Y->goi, 0 /* verbose_level */);

	cout << "Base_inv=" << endl;
	Y->D->print_matrix(Base_inv, Y->goi, Y->goi);


	cout << "T=" << endl;
	Y->D->print_matrix(T, N, N);
	cout << "Tv=" << endl;
	Y->D->print_matrix(Tv, N, N);




	INT *New_Base;
	INT *New_Base_inv;

	New_Base = NEW_INT(Y->goi * Y->goi * Y->D->size_of_instance_in_INT);
	New_Base_inv = NEW_INT(Y->goi * Y->goi * Y->D->size_of_instance_in_INT);

	Y->D->mult_matrix(Tv, Base, New_Base, N, N, N, 0);
	Y->D->mult_matrix(Base_inv, T, New_Base_inv, N, N, N, 0);

	cout << "New_Base=" << endl;
	Y->D->print_matrix(New_Base, N, N);
	cout << "New_Base_inv=" << endl;
	Y->D->print_matrix(New_Base_inv, N, N);


#if 0
	for (h = 0; h < Y->goi; h++) {
		
		Y->create_representation(New_Base, New_Base_inv, Y->goi, h, Rep2 + h * sz, verbose_level);

		cout << "Group element " << h << " is represented by" << endl;
		Y->D->print_matrix(Rep2 + h * sz, Y->goi, Y->goi);

		}
#endif

	cout << "before free" << endl;

	FREE_INT(T);
	FREE_INT(Tv);
	FREE_INT(I1);
	FREE_INT(I2);
	FREE_INT(Zero);
	FREE_INT(minus_Mu);
	FREE_INT(Mu);
	FREE_INT(Rep);
	FREE_INT(Rep2);
	FREE_INT(Rep3);
	FREE_INT(part);
	FREE_INT(parts);
	FREE_INT(Fst);
	FREE_INT(Len);
	cout << "before freeing Base" << endl;
	FREE_INT(Base);
	FREE_INT(Base_inv);
	cout << "before freeing Y" << endl;
	delete Y;
	cout << "before freeing elt1" << endl;
	FREE_INT(elt1);
	FREE_INT(elt2);
	FREE_INT(h_alpha);
	FREE_INT(elt4);
	FREE_INT(elt5);
	FREE_INT(elt6);
	FREE_INT(elt7);
#endif



