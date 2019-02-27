/*
 * action_group_theory.cpp
 *
 *  Created on: Feb 5, 2019
 *      Author: betten
 */

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {



void action::normalizer_using_MAGMA(
		const char *fname_magma_prefix,
		sims *G, sims *H, strong_generators *&gens_N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname_magma[1000];
	char fname_output[1000];
	char cmd[1000];

	if (f_v) {
		cout << "action::normalizer_using_MAGMA" << endl;
		}
	sprintf(fname_magma, "%s.magma", fname_magma_prefix);
	sprintf(fname_output, "%s.txt", fname_magma_prefix);

	int n;

	strong_generators *G_gen;
	strong_generators *H_gen;

	G_gen = NEW_OBJECT(strong_generators);
	G_gen->init_from_sims(G, 0 /* verbose_level */);

	H_gen = NEW_OBJECT(strong_generators);
	H_gen->init_from_sims(H, 0 /* verbose_level */);

	n = degree;
	if (f_v) {
		cout << "action::normalizer_using_MAGMA n = " << n << endl;
		}
	{
	ofstream fp(fname_magma);

	fp << "G := PermutationGroup< " << n << " | " << endl;
	G_gen->print_generators_MAGMA(this, fp);
	fp << ">;" << endl;


	fp << "H := sub< G |" << endl;
	H_gen->print_generators_MAGMA(this, fp);
	fp << ">;" << endl;

	fp << "N := Normalizer(G, H);" << endl;
	fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
	fp << "printf \"%o\", #N; printf \"\\n\";" << endl;
	fp << "printf \"%o\", #Generators(N); printf \"\\n\";" << endl;
	fp << "for h := 1 to #Generators(N) do for i := 1 to "
			<< n << " do printf \"%o\", i^N.h; printf \" \"; "
			"if i mod 25 eq 0 then printf \"\n\"; end if; "
			"end for; printf \"\\n\"; end for;" << endl;
	fp << "UnsetOutputFile();" << endl;
	}

	if (file_size(fname_output) > 0) {
		if (f_v) {
			cout << "file " << fname_output
					<< " exists, reading it" << endl;
		}

		int i, j;
		int go, nb_gens;
		int *perms;

		if (f_v) {
			cout << "action::normalizer_using_MAGMA" << endl;
			}
		{
			ifstream fp(fname_output);

			fp >> go;
			fp >> nb_gens;
			if (f_v) {
				cout << "action::normalizer_using_MAGMA We found " << nb_gens
						<< " generators for a group of order " << go << endl;
			}

			perms = NEW_int(nb_gens * degree);

			for (i = 0; i < nb_gens; i++) {
				for (j = 0; j < degree; j++) {
					fp >> perms[i * degree + j];
					}
				}
			if (f_v) {
				cout << "action::normalizer_using_MAGMA we read all "
						"generators from file " << fname_output << endl;
			}
		}
		for (i = 0; i < nb_gens * degree; i++) {
			perms[i]--;
			}

		//longinteger_object go1;


		gens_N = NEW_OBJECT(strong_generators);
		if (f_v) {
			cout << "action::normalizer_using_MAGMA "
				"before gens->init_from_permutation_"
				"representation" << endl;
		}

		vector_ge *nice_gens;

		gens_N->init_from_permutation_representation(this,
			perms,
			nb_gens, go, nice_gens,
			verbose_level);
		if (f_v) {
			cout << "action::normalizer_using_MAGMA "
				"after gens->init_from_permutation_"
				"representation" << endl;

		FREE_OBJECT(nice_gens);
		}


	} // if (file_size(fname_output))
	else {

		cout << "please exectute the magma file " << fname_magma
				<< " and come back" << endl;

		exit(0);
		sprintf(cmd, "/scratch/magma/magma %s", fname_magma);
		cout << "executing normalizer command in MAGMA" << endl;
		system(cmd);

		cout << "normalizer command in MAGMA has finished" << endl;
	}

	if (f_v) {
		cout << "action::normalizer_using_MAGMA done" << endl;
		}
}

void action::conjugacy_classes_using_MAGMA(const char *prefix,
		sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname_magma[1000];
	char fname_output[1000];
	char cmd[1000];

	if (f_v) {
		cout << "action::conjugacy_classes_using_MAGMA" << endl;
		}
	sprintf(fname_magma, "%sconjugacy_classes.magma", prefix);
	sprintf(fname_output, "%sconjugacy_classes.txt", prefix);

	int n;

	strong_generators *G_gen;

	G_gen = NEW_OBJECT(strong_generators);
	G_gen->init_from_sims(G, 0 /* verbose_level */);

	n = degree;
	if (f_v) {
		cout << "action::conjugacy_classes_using_MAGMA n = " << n << endl;
		}
	{
	ofstream fp(fname_magma);

	fp << "G := PermutationGroup< " << n << " | " << endl;
	G_gen->print_generators_MAGMA(this, fp);
	fp << ">;" << endl;


	fp << "C := ConjugacyClasses(G);" << endl;

	fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
	fp << "printf \"%o\", #C; printf \"\\n\";" << endl;
	fp << "for h := 1 to #C do  printf \"%o\", C[h][1]; printf \" \";"
			"printf \"%o\", C[h][2]; printf \" \";   for i := 1 to "
			<< n << " do printf \"%o\", i^C[h][3]; printf \" \"; end for; "
			"printf \"\\n\"; end for;" << endl;
	fp << "UnsetOutputFile();" << endl;
	}
	sprintf(cmd, "/scratch/magma/magma %s", fname_magma);
	cout << "executing ConjugacyClasses command in MAGMA" << endl;
	system(cmd);

	cout << "command ConjugacyClasses in MAGMA has finished" << endl;

	FREE_OBJECT(G_gen);





	if (f_v) {
		cout << "action::conjugacy_classes_using_MAGMA done" << endl;
		}
}

//G := PermutationGroup< 85 |  ...
//C := ConjugacyClasses(G);
//SetOutputFile("PGGL_4_4conjugacy_classes.txt");
//printf "%o", #C; printf "\n";
//for h := 1 to #C do  printf "%o", C[h][1]; printf " ";  printf "%o", C[h][2]; printf " ";   for i := 1 to 85 do printf "%o", i^C[h][3]; printf " "; end for; printf "\n"; end for;
//UnsetOutputFile();

// outputs:
//63
//1 1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85
//2 5355 1 2 9 26 45 6 7 8 3 11 10 13 12 15 14 17 16 19 18 21 20 23 22 25 24 4 30 29 28 27 34 33 32 31 38 37 36 35 41 42 39 40 44 43 5 48 49 46 47 52 53 50 51 55 54 57 56 59 58 61 60 63 62 65 64 67 66 69 68 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85
// etc.


void action::read_conjugacy_classes_from_MAGMA(
		char *fname,
		int &nb_classes,
		int *&perms,
		int *&class_size,
		int *&class_order_of_element,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "action::read_conjugacy_classes_from_MAGMA" << endl;
		}
	{
		ifstream fp(fname);

		fp >> nb_classes;
		if (f_v) {
			cout << "action::read_conjugacy_classes_from_MAGMA "
					"We found " << nb_classes
					<< " conjugacy classes" << endl;
		}

		perms = NEW_int(nb_classes * degree);
		class_size = NEW_int(nb_classes);
		class_order_of_element = NEW_int(nb_classes);

		for (i = 0; i < nb_classes; i++) {
			fp >> class_order_of_element[i];
			fp >> class_size[i];
			for (j = 0; j < degree; j++) {
				fp >> perms[i * degree + j];
				}
			}
		if (f_v) {
			cout << "action::read_conjugacy_classes_from_MAGMA "
					"we read all class representatives "
					"from file " << fname << endl;
		}
	}
	for (i = 0; i < nb_classes * degree; i++) {
		perms[i]--;
		}
	if (f_v) {
		cout << "action::read_conjugacy_classes_from_MAGMA done" << endl;
		}
}

void action::conjugacy_classes_and_normalizers_using_MAGMA(
		const char *prefix,
		sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname_magma[1000];
	char fname_output[1000];
	char cmd[1000];

	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA" << endl;
		}
	sprintf(fname_magma, "%s_classes.magma", prefix);
	sprintf(fname_output, "%s_classes_out.txt", prefix);

	int n;

	strong_generators *G_gen;

	G_gen = NEW_OBJECT(strong_generators);
	G_gen->init_from_sims(G, 0 /* verbose_level */);

	n = degree;
	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA n = " << n << endl;
		}
	{
	ofstream fp(fname_magma);

	fp << "G := PermutationGroup< " << n << " | " << endl;
	G_gen->print_generators_MAGMA(this, fp);
	fp << ">;" << endl;


	//fp << "# compute conjugacy classes of G:" << endl;
	fp << "C := ConjugacyClasses(G);" << endl;
	fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
	fp << "printf \"%o\", #C; printf \"\\n\";" << endl;
	fp << "for h := 1 to #C do" << endl;
	fp << "  printf \"%o\", C[h][1]; printf \" \";" << endl;
	fp << "  printf \"%o\", C[h][2]; printf \" \";" << endl;
	fp << "  for i := 1 to " << n << " do" << endl;
	fp << "    printf \"%o\", i^C[h][3]; printf \" \";" << endl;
	fp << "  end for; " << endl;
	fp << "  printf \"\\n\";" << endl;
	fp << "end for;" << endl;

	//fp << "# compute normalizers of the cyclic subgroups generated by the class reps:" << endl;
	fp << "for h := 1 to #C do" << endl;
	fp << "  S := sub< G | C[h][3]>;" << endl;
	fp << "  N := Normalizer(G, S);";
	fp << "  printf \"%o\", #N;" << endl;
	fp << "  printf \"\\n\";" << endl;
	fp << "  printf \"%o\", #Generators(N); printf \"\\n\";" << endl;
	fp << "  for g in Generators(N) do " << endl;
	fp << "    for i := 1 to " << n << " do " << endl;
	fp << "      printf \"%o\", i^g; printf \" \";" << endl;
	fp << "    end for;" << endl;
	fp << "    printf \"\\n\";" << endl;
	fp << "  end for;" << endl;
	fp << "end for;" << endl;
	fp << "UnsetOutputFile();" << endl;
	}
	sprintf(cmd, "/scratch/magma/magma %s", fname_magma);
	cout << "executing ConjugacyClasses command in MAGMA" << endl;
	system(cmd);

	cout << "command ConjugacyClasses in MAGMA has finished" << endl;

	FREE_OBJECT(G_gen);





	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA done" << endl;
		}
}


//> M24 := sub< Sym(24) |
//>  (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24),
//>  (2,16,9,6,8)(3,12,13,18,4)(7,17,10,11,22)(14,19,21,20,15),
//>  (1,22)(2,11)(3,15)(4,17)(5,9)(6,19)(7,13)(8,20)(10,16)(12,21)(14,18)(23,24)>;
//> M24;


void action::read_conjugacy_classes_and_normalizers_from_MAGMA(
		char *fname,
		int &nb_classes,
		int *&perms,
		int *&class_size,
		int *&class_order_of_element,
		int *&class_normalizer_order,
		int *&class_normalizer_number_of_generators,
		int **&normalizer_generators_perms,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;

	if (f_v) {
		cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
		}
	{
		ifstream fp(fname);

		fp >> nb_classes;
		if (f_v) {
			cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA "
					"We found " << nb_classes
					<< " conjugacy classes" << endl;
		}

		perms = NEW_int(nb_classes * degree);
		class_size = NEW_int(nb_classes);
		class_order_of_element = NEW_int(nb_classes);

		for (i = 0; i < nb_classes; i++) {
			fp >> class_order_of_element[i];
			fp >> class_size[i];
			for (j = 0; j < degree; j++) {
				fp >> perms[i * degree + j];
			}
		}
		if (f_v) {
			cout << "perms:" << endl;
			int_matrix_print(perms, nb_classes, degree);
		}
		for (i = 0; i < nb_classes * degree; i++) {
			perms[i]--;
		}

		class_normalizer_order = NEW_int(nb_classes);
		class_normalizer_number_of_generators = NEW_int(nb_classes);
		normalizer_generators_perms = NEW_pint(nb_classes);

		for (i = 0; i < nb_classes; i++) {
			fp >> class_normalizer_order[i];
			fp >> class_normalizer_number_of_generators[i];
			normalizer_generators_perms[i] =
					NEW_int(class_normalizer_number_of_generators[i] * degree);
			for (h = 0; h < class_normalizer_number_of_generators[i]; h++) {
				for (j = 0; j < degree; j++) {
					fp >> normalizer_generators_perms[i][h * degree + j];
				}
			}
			for (h = 0; h < class_normalizer_number_of_generators[i] * degree; h++) {
				normalizer_generators_perms[i][h]--;
			}
		}
		if (f_v) {
			cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA "
					"we read all class representatives "
					"from file " << fname << endl;
		}
	}
	if (f_v) {
		cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA done" << endl;
		}
}



void action::centralizer_using_MAGMA(const char *prefix,
		sims *G, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname_magma[1000];
	char fname_output[1000];
	char cmd[1000];

	if (f_v) {
		cout << "action::centralizer_using_MAGMA" << endl;
		}
	sprintf(fname_magma, "%scentralizer.magma", prefix);
	sprintf(fname_output, "%scentralizer.txt", prefix);

	int n;

	strong_generators *G_gen;

	G_gen = NEW_OBJECT(strong_generators);
	G_gen->init_from_sims(G, 0 /* verbose_level */);

	n = degree;
	if (f_v) {
		cout << "action::centralizer_using_MAGMA n = " << n << endl;
		}
	{
	ofstream fp(fname_magma);

	fp << "G := PermutationGroup< " << n << " | " << endl;
	G_gen->print_generators_MAGMA(this, fp);
	fp << ">;" << endl;

	fp << "h := G ! ";
	element_print_as_permutation_with_offset(Elt, fp,
			1 /* offset */, TRUE /* f_do_it_anyway_even_for_big_degree */,
			FALSE /* f_print_cycles_of_length_one */, 0 /* verbose_level */);
	fp << ";" << endl;

	fp << "C := Centralizer(G, h);" << endl;

	fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
	fp << "printf \"%o\", #C; printf \"\\n\";" << endl;
	fp << "printf \"%o\", #Generators(C); printf \"\\n\";" << endl;
	fp << "for h := 1 to #Generators(C) do for i := 1 to "
			<< n << " do printf \"%o\", i^C.h; printf \" \"; end for;"
			" printf \"\\n\"; end for;" << endl;
	fp << "UnsetOutputFile();" << endl;
	}
	cout << "Written file " << fname_magma
			<< " of size " << file_size(fname_magma) << endl;

	sprintf(cmd, "/scratch/magma/magma %s", fname_magma);
	cout << "executing centralizer command in MAGMA" << endl;
	system(cmd);

	cout << "command centralizer in MAGMA has finished" << endl;

	FREE_OBJECT(G_gen);

	if (f_v) {
		cout << "action::centralizer_using_MAGMA done" << endl;
		}
}




void action::conjugacy_classes_and_normalizers(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char prefix[1000];
	char fname1[1000];
	char fname2[1000];


	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers" << endl;
		}

	sprintf(prefix, "%s", group_prefix);
	sprintf(fname1, "%s_classes.magma", prefix);
	sprintf(fname2, "%s_classes_out.txt", prefix);


	if (file_size(fname2) > 0) {
		read_conjugacy_classes_and_normalizers(fname2, verbose_level);
		}
	else {
		conjugacy_classes_and_normalizers_using_MAGMA(prefix,
				Sims, verbose_level);
		}

	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers done" << endl;
		}
}


void action::read_conjugacy_classes_and_normalizers(
		char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_classes;
	int *perms;
	int *class_size;
	int *class_order_of_element;
	int *class_normalizer_order;
	int *class_normalizer_number_of_generators;
	int **normalizer_generators_perms;
	//projective_space_with_action *PA;

	if (f_v) {
		cout << "action::read_conjugacy_classes_and_normalizers" << endl;
		}

	read_conjugacy_classes_and_normalizers_from_MAGMA(
			fname,
			nb_classes,
			perms,
			class_size,
			class_order_of_element,
			class_normalizer_order,
			class_normalizer_number_of_generators,
			normalizer_generators_perms,
			verbose_level - 1);


#if 0
	PA = NEW_OBJECT(projective_space_with_action);

	int f_semilinear;

	if (is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}
	PA->init(
		F, 3 /* n */, f_semilinear,
		FALSE /* f_init_incidence_structure */,
		verbose_level);
#endif


	longinteger_object go;
	longinteger_domain D;

	group_order(go);
	cout << "The group has order " << go << endl;

	char fname_latex[1000];
	strcpy(fname_latex, fname);

	replace_extension_with(fname_latex, ".tex");

	{
	ofstream fp(fname_latex);
	char title[1000];

	sprintf(title, "Conjugacy classes of $%s$",
			label_tex);

	latex_head(fp,
		FALSE /* f_book */, TRUE /* f_title */,
		title, "computed by Orbiter and MAGMA" /* const char *author */,
		FALSE /* f_toc */, FALSE /* f_landscape */, TRUE /* f_12pt */,
		TRUE /* f_enlarged_page */, TRUE /* f_pagenumbers */,
		NULL /* extra_praeamble */);
	//latex_head_easy(fp);

	fp << "\\section{Conjugacy classes in $"
			<< label_tex << "$}" << endl;


	fp << "The group order is " << endl;
	fp << "$$" << endl;
	go.print_not_scientific(fp);
	fp << endl;
	fp << "$$" << endl;


	cout << "The conjugacy classes are:" << endl;
	for (i = 0; i < nb_classes; i++) {
		strong_generators *gens;
		longinteger_object go1, Class_size, centralizer_order;
		int goi;
		vector_ge *nice_gens;


		goi = class_order_of_element[i];
		gens = NEW_OBJECT(strong_generators);

		gens->init_from_permutation_representation(this,
			perms + i * degree,
			1, goi, nice_gens,
			verbose_level);

		if (f_v) {
			cout << "action::normalizer_using_MAGMA "
				"after gens->init_from_permutation_"
				"representation" << endl;
		}

		Class_size.create(class_size[i]);

		D.integral_division_exact(go, Class_size, centralizer_order);



		int ngo;
		int nb_perms;
		strong_generators *N_gens;
		vector_ge *nice_gens_N;

		ngo = class_normalizer_order[i];
		nb_perms = class_normalizer_number_of_generators[i];

		//int *class_normalizer_order;
		//int *class_normalizer_number_of_generators;
		//int **normalizer_generators_perms;

		N_gens = NEW_OBJECT(strong_generators);
		N_gens->init_from_permutation_representation(this,
				normalizer_generators_perms[i],
				nb_perms, ngo, nice_gens_N,
				verbose_level - 1);

		cout << "class " << i << " / " << nb_classes
			<< " size = " << class_size[i]
			<< " order of element = " << class_order_of_element[i]
			<< " centralizer order = " << centralizer_order
			<< " normalizer order = " << ngo
			<< " : " << endl;
		cout << "packing::read_conjugacy_classes_and_normalizers created "
				"generators for a group" << endl;
		gens->print_generators();
		gens->print_generators_as_permutations();
		gens->group_order(go1);
		cout << "packing::read_conjugacy_classes_and_normalizers "
				"The group has order " << go1 << endl;

		fp << "\\bigskip" << endl;
		fp << "\\subsection*{Class " << i << " / "
				<< nb_classes << "}" << endl;
		fp << "Order of element = " << class_order_of_element[i]
				<< "\\\\" << endl;
		fp << "Class size = " << class_size[i] << "\\\\" << endl;
		fp << "Centralizer order = " << centralizer_order
				<< "\\\\" << endl;
		fp << "Normalizer order = " << ngo
				<< "\\\\" << endl;

		int *Elt = NULL;


		if (class_order_of_element[i] > 1) {
			Elt = nice_gens->ith(0);
			fp << "Representing element is" << endl;
			fp << "$$" << endl;
			element_print_latex(Elt, fp);
			fp << "$$" << endl;
			fp << "$";
			element_print_for_make_element(Elt, fp);
			fp << "$\\\\" << endl;



		}
		fp << "The normalizer is generated by:\\\\" << endl;
		N_gens->print_generators_tex(fp);


#if 0
		if (class_order_of_element[i] > 1) {
			fp << "The fix structure is:\\\\" << endl;
			PA->report_fixed_objects_in_PG_3_tex(
					Elt, fp,
					verbose_level);

			fp << "The orbit structure is:\\\\" << endl;
			PA->report_orbits_in_PG_3_tex(
				Elt, fp,
				verbose_level);
		}
		if (class_order_of_element[i] > 1) {

			PA->report_decomposition_by_single_automorphism(
					Elt, fp,
					verbose_level);
			// PA->
			//action *A; // linear group PGGL(d,q)
			//action *A_on_lines; // linear group PGGL(d,q) acting on lines


		}
#endif

		FREE_int(normalizer_generators_perms[i]);

		FREE_OBJECT(nice_gens_N);
		FREE_OBJECT(nice_gens);
		FREE_OBJECT(N_gens);
		FREE_OBJECT(gens);
		}
	latex_foot(fp);
	}
	cout << "Written file " << fname_latex << " of size "
			<< file_size(fname_latex) << endl;

	FREE_int(perms);
	FREE_int(class_size);
	FREE_int(class_order_of_element);
	FREE_int(class_normalizer_order);
	FREE_int(class_normalizer_number_of_generators);
	FREE_pint(normalizer_generators_perms);
	//FREE_OBJECT(PA);

	if (f_v) {
		cout << "action::read_conjugacy_classes_and_normalizers done" << endl;
		}
}


void action::report_fixed_objects(int *Elt,
		char *fname_latex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, j, cnt;
	//int v[4];

	if (f_v) {
		cout << "action::report_fixed_objects" << endl;
		}


	{
	ofstream fp(fname_latex);
	char title[1000];

	sprintf(title, "Fixed Objects");

	latex_head(fp,
		FALSE /* f_book */, TRUE /* f_title */,
		title, "" /* const char *author */,
		FALSE /* f_toc */, FALSE /* f_landscape */, TRUE /* f_12pt */,
		TRUE /* f_enlarged_page */, TRUE /* f_pagenumbers */,
		NULL /* extra_praeamble */);
	//latex_head_easy(fp);

	fp << "\\section{Fixed Objects}" << endl;



	fp << "The element" << endl;
	fp << "$$" << endl;
	element_print_latex(Elt, fp);
	fp << "$$" << endl;
	fp << "has the following fixed objects:" << endl;


#if 0
	fp << "\\subsection{Fixed Points}" << endl;

	cnt = 0;
	for (i = 0; i < P3->N_points; i++) {
		j = element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
			}
		}

	fp << "There are " << cnt << " fixed points, they are: \\\\" << endl;
	for (i = 0; i < P3->N_points; i++) {
		j = element_image_of(i, Elt, 0 /* verbose_level */);
		F->PG_element_unrank_modified(v, 1, 4, i);
		if (j == i) {
			fp << i << " : ";
			int_vec_print(fp, v, 4);
			fp << "\\\\" << endl;
			cnt++;
			}
		}

	fp << "\\subsection{Fixed Lines}" << endl;

	{
	action *A2;

	A2 = induced_action_on_grassmannian(2, 0 /* verbose_level*/);

	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
			}
		}

	fp << "There are " << cnt << " fixed lines, they are: \\\\" << endl;
	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			fp << i << " : $\\left[";
			A2->G.AG->G->print_single_generator_matrix_tex(fp, i);
			fp << "\\right]$\\\\" << endl;
			cnt++;
			}
		}

	FREE_OBJECT(A2);
	}

	fp << "\\subsection{Fixed Planes}" << endl;

	{
	action *A2;

	A2 = induced_action_on_grassmannian(3, 0 /* verbose_level*/);

	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			cnt++;
			}
		}

	fp << "There are " << cnt << " fixed planes, they are: \\\\" << endl;
	cnt = 0;
	for (i = 0; i < A2->degree; i++) {
		j = A2->element_image_of(i, Elt, 0 /* verbose_level */);
		if (j == i) {
			fp << i << " : $\\left[";
			A2->G.AG->G->print_single_generator_matrix_tex(fp, i);
			fp << "\\right]$\\\\" << endl;
			cnt++;
			}
		}

	FREE_OBJECT(A2);
	}
#endif


	latex_foot(fp);
	}
	cout << "Written file " << fname_latex << " of size "
			<< file_size(fname_latex) << endl;


	if (f_v) {
		cout << "action::report_fixed_objects done" << endl;
		}
}




}}

