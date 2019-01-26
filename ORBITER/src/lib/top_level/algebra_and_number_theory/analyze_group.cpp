// analyze_group.C
// 
// Anton Betten
// started:     03/10/2009
// last change: 03/12/2009
//
// 
//
//

#include "orbiter.h"

using namespace orbiter::foundations;

namespace orbiter {

void analyze_group(action *A, sims *S,
		vector_ge *SG, vector_ge *gens2, int verbose_level)
{
	int *Elt1;
	int *Elt2;
	int i, goi;
	longinteger_object go;
	int *perm;
	int *primes;
	int *exponents;
	int factorization_length;
	int nb_primes, nb_gens2;
	
	
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	
	
	S->group_order(go);
	goi = go.as_int();

	factorization_length = factor_int(goi, primes, exponents);
	cout << "analyzing a group of order " << goi << " = ";
	print_factorization(factorization_length, primes, exponents);
	cout << endl;
	
	nb_primes = 0;
	for (i = 0; i < factorization_length; i++) {
		nb_primes += exponents[i];
		}
	cout << "nb_primes=" << nb_primes << endl;
	gens2->init(A);
	gens2->allocate(nb_primes);

	compute_regular_representation(A, S, SG, perm, verbose_level);

	int *center;
	int size_center;
	
	center = NEW_int(goi);
	
	S->center(*SG, center, size_center, verbose_level);
	
	cout << "the center is:" << endl;
	for (i = 0; i < size_center; i++) {
		cout << i << " element has rank " << center[i] << endl;
		S->element_unrank_int(center[i], Elt1);
		A->print(cout, Elt1);
		//A->print_as_permutation(cout, Elt1);
		cout << endl;
		}
	cout << endl << endl;
	
	S->element_unrank_int(center[1], Elt1);
	A->move(Elt1, gens2->ith(0));
	nb_gens2 = 1;
	
	cout << "chosen generator " << nb_gens2 - 1 << endl;
	A->print(cout, gens2->ith(nb_gens2 - 1));
	
	factor_group *FactorGroup;

	FactorGroup = NEW_OBJECT(factor_group);
		
	create_factor_group(A, S, goi, size_center, center,
			FactorGroup, verbose_level);
	
	cout << "FactorGroup created" << endl;
	cout << "Order of FactorGroup is " <<
			FactorGroup->goi_factor_group << endl;
	

	cout << "computing the regular representation of degree "
			<< FactorGroup->goi_factor_group << ":" << endl;
	
	
	for (i = 0; i < SG->len; i++) {
		FactorGroup->FactorGroup->print_as_permutation(cout, SG->ith(i));
		cout << endl;
		}
	cout << endl;


#if 0
	cout << "now listing all elements:" << endl;
	for (i = 0; i < FactorGroup->goi_factor_group; i++) {
		FactorGroup->FactorGroup->Sims->element_unrank_int(i, Elt1);
		cout << "element " << i << ":" << endl;
		A->print(cout, Elt1);
		FactorGroup->FactorGroupConjugated->print_as_permutation(cout, Elt1);
		cout << endl;
		}
	cout << endl << endl;
#endif




	sims H1, H2, H3;
	longinteger_object goH1, goH2, goH3;
	vector_ge SGH1, SGH2, SGH3;
	int *tl1, *tl2, *tl3, *tlF1, *tlF2;
	
	tl1 = NEW_int(A->base_len);
	tl2 = NEW_int(A->base_len);
	tl3 = NEW_int(A->base_len);
	tlF1 = NEW_int(A->base_len);
	tlF2 = NEW_int(A->base_len);
	
	
	// now we compute H1, the derived group
	
	
	H1.init(FactorGroup->FactorGroup);
	H1.init_trivial_group(verbose_level - 1);
	H1.build_up_subgroup_random_process(FactorGroup->FactorGroup->Sims, 
		choose_random_generator_derived_group, verbose_level - 1);
	H1.group_order(goH1);
	cout << "the commutator subgroup has order " << goH1 << endl << endl;
	H1.extract_strong_generators_in_order(SGH1, tl1, verbose_level - 2);
	for (i = 0; i < SGH1.len; i++) {
		cout << "generator " << i << ":" << endl;
		A->print(cout, SGH1.ith(i));
		//cout << "as permutation in FactorGroupConjugated:" << endl;
		//FactorGroup->FactorGroupConjugated->print_as_permutation(
		// cout, SGH1.ith(i));
		//cout << endl;
		}
	cout << endl << endl;
	

	int size_H1;
	int *elts_H1;
	
	size_H1 = goH1.as_int();
	elts_H1 = NEW_int(size_H1);


	FactorGroup->FactorGroup->Sims->element_ranks_subgroup(
			&H1, elts_H1, verbose_level);
	cout << "the ranks of elements in H1 are:" << endl;
	int_vec_print(cout, elts_H1, size_H1);
	cout << endl;

	factor_group *ModH1;

	ModH1 = NEW_OBJECT(factor_group);

	create_factor_group(FactorGroup->FactorGroupConjugated, 
		FactorGroup->FactorGroup->Sims, 
		FactorGroup->goi_factor_group, 
		size_H1, elts_H1, ModH1, verbose_level);
		

	
	cout << "ModH1 created" << endl;
	cout << "Order of ModH1 is " << ModH1->goi_factor_group << endl;



	cout << "the elements of ModH1 are:" << endl;
	for (i = 0; i < ModH1->goi_factor_group; i++) {
		cout << "element " << i << ":" << endl;
		ModH1->FactorGroup->Sims->element_unrank_int(i, Elt1);
		A->print(cout, Elt1);
		cout << endl;
		cout << "in the factor group mod H1" << endl;
		ModH1->FactorGroupConjugated->print_as_permutation(cout, Elt1);
		cout << endl;
		cout << "in the factor group mod center" << endl;
		FactorGroup->FactorGroupConjugated->print_as_permutation(
				cout, Elt1);
		cout << endl;
		}




	// now we compute H2, the second derived group
	
	
	H2.init(FactorGroup->FactorGroup);
	H2.init_trivial_group(verbose_level - 1);
	H2.build_up_subgroup_random_process(&H1, 
		choose_random_generator_derived_group, verbose_level - 1);
	H2.group_order(goH2);
	cout << "the second commutator subgroup has order "
			<< goH2 << endl << endl;
	H2.extract_strong_generators_in_order(SGH2, tl2, verbose_level - 2);
	for (i = 0; i < SGH2.len; i++) {
		cout << "generator " << i << ":" << endl;
		A->print(cout, SGH2.ith(i));
		//cout << "as permutation in FactorGroupConjugated:" << endl;
		//FactorGroup->FactorGroupConjugated->print_as_permutation(
		// cout, SGH2.ith(i));
		//cout << endl;
		
		A->move(SGH2.ith(i), gens2->ith(nb_gens2));
		nb_gens2++;
		cout << "chosen generator " << nb_gens2 - 1 << endl;
		A->print(cout, gens2->ith(nb_gens2 - 1));

		}
	cout << endl << endl;
	
	int size_H2;
	int *elts_H2;
	
	size_H2 = goH2.as_int();
	elts_H2 = NEW_int(size_H1);


	H1.element_ranks_subgroup(&H2, elts_H2, verbose_level);
	cout << "the ranks of elements in H2 are:" << endl;
	int_vec_print(cout, elts_H2, size_H2);
	cout << endl;

	factor_group *ModH2;

	ModH2 = NEW_OBJECT(factor_group);

	create_factor_group(FactorGroup->FactorGroupConjugated, 
		&H1, 
		size_H1, 
		size_H2, elts_H2, ModH2, verbose_level);
		

	
	cout << "ModH2 created" << endl;
	cout << "Order of ModH2 is " << ModH2->goi_factor_group << endl;

	cout << "the elements of ModH2 are:" << endl;
	for (i = 0; i < ModH2->goi_factor_group; i++) {
		cout << "element " << i << ":" << endl;
		ModH2->FactorGroup->Sims->element_unrank_int(i, Elt1);
		A->print(cout, Elt1);
		cout << endl;
		cout << "in the factor group mod H2" << endl;
		ModH2->FactorGroupConjugated->print_as_permutation(cout, Elt1);
		cout << endl;
		//cout << "in the factor group mod center" << endl;
		//FactorGroup->FactorGroupConjugated->print_as_permutation(
		// cout, Elt1);
		//cout << endl;
		}
	
	vector_ge SG_F1, SG_F2;
	
	ModH2->FactorGroup->Sims->extract_strong_generators_in_order(
			SG_F2, tlF2, verbose_level - 2);
	for (i = 0; i < SG_F2.len; i++) {
		cout << "generator " << i << " for ModH2:" << endl;
		A->print(cout, SG_F2.ith(i));
		//cout << "as permutation in FactorGroupConjugated:" << endl;
		//FactorGroup->FactorGroupConjugated->print_as_permutation(
		// cout, SGH2.ith(i));
		//cout << endl;
		
		A->move(SG_F2.ith(i), gens2->ith(nb_gens2));
		nb_gens2++;
		cout << "chosen generator " << nb_gens2 - 1 << endl;
		A->print(cout, gens2->ith(nb_gens2 - 1));

		}
	cout << endl << endl;

	ModH1->FactorGroup->Sims->extract_strong_generators_in_order(
			SG_F1, tlF1, verbose_level - 2);
	for (i = 0; i < SG_F1.len; i++) {
		cout << "generator " << i << " for ModH1:" << endl;
		A->print(cout, SG_F1.ith(i));
		//cout << "as permutation in FactorGroupConjugated:" << endl;
		//FactorGroup->FactorGroupConjugated->print_as_permutation(
		// cout, SGH2.ith(i));
		//cout << endl;
		
		A->move(SG_F1.ith(i), gens2->ith(nb_gens2));
		nb_gens2++;
		cout << "chosen generator " << nb_gens2 - 1 << endl;
		A->print(cout, gens2->ith(nb_gens2 - 1));

		}
	cout << endl << endl;

	cout << "we found " << nb_gens2 << " generators:" << endl;
	for (i = 0; i < nb_gens2; i++) {
		cout << "generator " << i << ":" << endl;
		A->print(cout, gens2->ith(i));
		}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(perm);
	FREE_int(tl1);
	FREE_int(tl2);
	FREE_int(tl3);
	FREE_int(tlF1);
	FREE_int(tlF2);
}

void compute_regular_representation(action *A, sims *S,
		vector_ge *SG, int *&perm, int verbose_level)
{
	longinteger_object go;
	int goi, i;
	
	S->group_order(go);
	goi = go.as_int();
	cout << "computing the regular representation of degree "
			<< go << ":" << endl;
	perm = NEW_int(SG->len * goi);
	
	for (i = 0; i < SG->len; i++) {
		S->regular_representation(SG->ith(i),
				perm + i * goi, verbose_level);
		}
	cout << endl;
	for (i = 0; i < SG->len; i++) {
		perm_print_offset(cout,
			perm + i * goi, goi, 1 /* offset */,
			FALSE /* f_cycle_length */, FALSE, 0,
			TRUE /* f_orbit_structure */);
		cout << endl;
		}
}

void presentation(action *A, sims *S, int goi,
		vector_ge *gens, int *primes, int verbose_level)
{
	int *Elt1, *Elt2, *Elt3, *Elt4;
	int i, j, jj, k, l, a, b;
	int word[100];
	int *word_list;
	int *inverse_word_list;
	
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	
	word_list = NEW_int(goi);
	inverse_word_list = NEW_int(goi);
	
	l = gens->len;
	
	cout << "presentation of length " << l << endl;
	cout << "primes: ";
	int_vec_print(cout, primes, l);
	cout << endl;
	
#if 0
	// replace g5 by  g5 * g3:
	A->mult(gens->ith(5), gens->ith(3), Elt1);
	A->move(Elt1, gens->ith(5));
	
	// replace g7 by  g7 * g4:
	A->mult(gens->ith(7), gens->ith(4), Elt1);
	A->move(Elt1, gens->ith(7));
#endif
	
	
	
	for (i = 0; i < goi; i++) {
		inverse_word_list[i] = -1;
		}
	for (i = 0; i < goi; i++) {
		A->one(Elt1);
		j = i;
		for (k = 0; k < l; k++) {
			b = j % primes[k];
			word[k] = b;
			j = j - b;
			j = j / primes[k];
			}
		for (k = 0; k < l; k++) {
			b = word[k];
			while (b) {
				A->mult(Elt1, gens->ith(k), Elt2);
				A->move(Elt2, Elt1);
				b--;
				}
			}
		A->move(Elt1, Elt2);
		a = S->element_rank_int(Elt2);
		word_list[i] = a;
		inverse_word_list[a] = i;
		cout << "word " << i << " = ";
		int_vec_print(cout, word, 9);
		cout << " gives " << endl;
		A->print(cout, Elt1);
		cout << "which is element " << word_list[i] << endl;
		cout << endl;
		}
	cout << "i : word_list[i] : inverse_word_list[i]" << endl;
	for (i = 0; i < goi; i++) {
		cout << setw(5) << i << " : " << setw(5) << word_list[i]
			<< " : " << setw(5) << inverse_word_list[i] << endl;
		}



	for (i = 0; i < l; i++) {
		cout << "generator " << i << ":" << endl;
		A->print(cout, gens->ith(i));
		cout << endl;
		}
	for (i = 0; i < l; i++) {
		A->move(gens->ith(i), Elt1);
		A->element_power_int_in_place(Elt1, primes[i], 0);
		a = S->element_rank_int(Elt1);
		cout << "generator " << i << " to the power " << primes[i]
			<< " is elt " << a << " which is word "
			<< inverse_word_list[a];
		j = inverse_word_list[a];
		for (k = 0; k < l; k++) {
			b = j % primes[k];
			word[k] = b;
			j = j - b;
			j = j / primes[k];
			}
		int_vec_print(cout, word, l);
		cout << " :" << endl;
		A->print(cout, Elt1);
		cout << endl;
		}


	for (i = 0; i < l; i++) {
		A->move(gens->ith(i), Elt1);
		A->invert(Elt1, Elt2);
		for (j = 0; j < i; j++) {
			A->mult(Elt2, gens->ith(j), Elt3);
			A->mult(Elt3, Elt1, Elt4);
			cout << "g_" << j << "^{g_" << i << "} =" << endl;
			a = S->element_rank_int(Elt4);
			cout << "which is element " << a << " which is word "
				<< inverse_word_list[a] << " = ";
			jj = inverse_word_list[a];
			for (k = 0; k < l; k++) {
				b = jj % primes[k];
				word[k] = b;
				jj = jj - b;
				jj = jj / primes[k];
				}
			int_vec_print(cout, word, l);
			cout << endl;
			A->print(cout, Elt4);
			cout << endl;
			}
		cout << endl;
		}

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
	
	FREE_int(word_list);
	FREE_int(inverse_word_list);
}


}
