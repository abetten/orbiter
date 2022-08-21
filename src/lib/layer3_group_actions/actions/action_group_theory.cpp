/*
 * action_group_theory.cpp
 *
 *  Created on: Feb 5, 2019
 *      Author: betten
 */

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {



void action::normalizer_using_MAGMA(
		std::string &fname_magma_prefix,
		groups::sims *G, groups::sims *H, groups::strong_generators *&gens_N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_magma;
	string fname_output;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "action::normalizer_using_MAGMA" << endl;
	}

	fname_magma.assign(fname_magma_prefix);
	fname_magma.append(".magma");
	fname_output.assign(fname_magma_prefix);
	fname_output.append(".txt");

	int n;

	groups::strong_generators *G_gen;
	groups::strong_generators *H_gen;

	G_gen = NEW_OBJECT(groups::strong_generators);
	G_gen->init_from_sims(G, 0 /* verbose_level */);

	H_gen = NEW_OBJECT(groups::strong_generators);
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

	if (Fio.file_size(fname_output) <= 0) {

		orbiter_kernel_system::magma_interface Magma;

		Magma.run_magma_file(fname_magma, verbose_level);
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << Magma.Orbiter_session->magma_path << "magma " << fname_magma << endl;
		exit(1);
	}

	if (f_v) {
		cout << "file " << fname_output << " exists, reading it" << endl;
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


	gens_N = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "action::normalizer_using_MAGMA "
			"before gens->init_from_permutation_"
			"representation" << endl;
	}

	data_structures_groups::vector_ge *nice_gens;

	gens_N->init_from_permutation_representation(this, G,
		perms,
		nb_gens, go, nice_gens,
		verbose_level);
	if (f_v) {
		cout << "action::normalizer_using_MAGMA "
			"after gens->init_from_permutation_"
			"representation" << endl;
	}
	FREE_OBJECT(nice_gens);

	cout << "action::normalizer_using_MAGMA "
		"after gens->init_from_permutation_representation gens_N=" << endl;
	gens_N->print_generators_for_make_element(cout);




	if (f_v) {
		cout << "action::normalizer_using_MAGMA done" << endl;
	}
}

void action::conjugacy_classes_using_MAGMA(std::string &prefix,
		groups::sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_magma;
	string fname_output;

	if (f_v) {
		cout << "action::conjugacy_classes_using_MAGMA" << endl;
		}

	fname_magma.assign(prefix);
	fname_magma.append("conjugacy_classes.magma");
	fname_output.assign(prefix);
	fname_output.append("conjugacy_classes.txt");

	int n;

	groups::strong_generators *G_gen;

	G_gen = NEW_OBJECT(groups::strong_generators);
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



	orbiter_kernel_system::magma_interface Magma;
	orbiter_kernel_system::file_io Fio;

	Magma.run_magma_file(fname_magma, verbose_level);
	if (Fio.file_size(fname_output) == 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << Magma.Orbiter_session->magma_path << "magma " << fname_magma << endl;
		exit(1);
	}


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


#if 0
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
#endif

void action::conjugacy_classes_and_normalizers_using_MAGMA_make_fnames(
		std::string &prefix,
		std::string &fname_magma, std::string &fname_output)
{
	fname_magma.assign(prefix);
	fname_magma.append("_classes.magma");
	fname_output.assign(prefix);
	fname_output.append("_classes_out.txt");
}

void action::conjugacy_classes_and_normalizers_using_MAGMA(
		std::string &prefix,
		groups::sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_magma;
	string fname_output;

	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA" << endl;
	}
	conjugacy_classes_and_normalizers_using_MAGMA_make_fnames(prefix, fname_magma, fname_output);
	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA, fname_magma = " << fname_magma << endl;
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA, fname_output = " << fname_output << endl;
	}

	int n;

	groups::strong_generators *G_gen;

	G_gen = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA before G_gen->init_from_sims" << endl;
	}
	G_gen->init_from_sims(G, verbose_level);
	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA after G_gen->init_from_sims" << endl;
	}

	n = degree;
	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA n = " << n << endl;
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA fname_magma = " << fname_magma << endl;
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA fname_output = " << fname_output << endl;
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


	orbiter_kernel_system::magma_interface Magma;
	orbiter_kernel_system::file_io Fio;

	Magma.run_magma_file(fname_magma, verbose_level);
	if (Fio.file_size(fname_output) <= 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << Magma.Orbiter_session->magma_path << "magma " << fname_magma << endl;
		exit(1);
	}


	if (f_v) {
		cout << "command ConjugacyClasses in MAGMA has finished" << endl;
	}

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
		std::string &fname,
		int &nb_classes,
		int *&perms,
		int *&class_size,
		int *&class_order_of_element,
		long int *&class_normalizer_order,
		int *&class_normalizer_number_of_generators,
		int **&normalizer_generators_perms,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;

	if (f_v) {
		cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
		cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA fname=" << fname << endl;
		cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA degree=" << degree << endl;
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
			if (f_v) {
				cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA "
						"class " << i << " / " << nb_classes << " order=" << class_order_of_element[i] << endl;
			}
			fp >> class_size[i];
			for (j = 0; j < degree; j++) {
				fp >> perms[i * degree + j];
			}
		}
		if (f_v) {
			cout << "perms:" << endl;
			Int_matrix_print(perms, nb_classes, degree);
		}
		for (i = 0; i < nb_classes * degree; i++) {
			perms[i]--;
		}

		class_normalizer_order = NEW_lint(nb_classes);
		class_normalizer_number_of_generators = NEW_int(nb_classes);
		normalizer_generators_perms = NEW_pint(nb_classes);

		if (f_v) {
			cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA "
					"reading normalizer generators:" << endl;
		}
		for (i = 0; i < nb_classes; i++) {
			if (f_v) {
				cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA "
						"class " << i << " / " << nb_classes << endl;
			}
			fp >> class_normalizer_order[i];

			cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA class " << i << " class_normalizer_order[i]=" << class_normalizer_order[i] << endl;

			if (class_normalizer_order[i] <= 0) {
				cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA class_normalizer_order[i] <= 0" << endl;
				cout << "class_normalizer_order[i]=" << class_normalizer_order[i] << endl;
				exit(1);
			}
			if (f_v) {
				cout << "action::read_conjugacy_classes_and_normalizers_from_MAGMA "
						"class " << i << " / " << nb_classes << " class_normalizer_order[i]=" << class_normalizer_order[i] << endl;
			}
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


void action::normalizer_of_cyclic_group_using_MAGMA(std::string &fname_magma_prefix,
		groups::sims *G, int *Elt, groups::strong_generators *&gens_N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::normalizer_of_cyclic_group_using_MAGMA" << endl;
	}
	groups::sims *H;

	H = NEW_OBJECT(groups::sims);
	if (f_v) {
		cout << "action::normalizer_of_cyclic_group_using_MAGMA "
				"before H->init_cyclic_group_from_generator" << endl;
	}
	H->init_cyclic_group_from_generator(G->A, Elt, verbose_level);
	if (f_v) {
		cout << "action::normalizer_of_cyclic_group_using_MAGMA "
				"after H->init_cyclic_group_from_generator" << endl;
	}

	if (f_v) {
		cout << "action::normalizer_of_cyclic_group_using_MAGMA "
				"before normalizer_using_MAGMA" << endl;
	}
	normalizer_using_MAGMA(
		fname_magma_prefix,
		G, H, gens_N,
		verbose_level);
	if (f_v) {
		cout << "action::normalizer_of_cyclic_group_using_MAGMA "
				"after normalizer_using_MAGMA" << endl;
	}

	if (f_v) {
		cout << "action::normalizer_of_cyclic_group_using_MAGMA done" << endl;
	}
}

void action::centralizer_using_MAGMA(std::string &prefix,
		groups::sims *override_Sims, int *Elt, groups::strong_generators *&gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_magma;
	string fname_output;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "action::centralizer_using_MAGMA" << endl;
	}

	fname_magma.assign(prefix);
	fname_magma.append("_centralizer.magma");
	fname_output.assign(prefix);
	fname_output.append("_centralizer.txt");


	if (Fio.file_size(fname_output) > 0) {
		read_centralizer_magma(fname_output, override_Sims,
				gens, verbose_level);
	}
	else {
		if (f_v) {
			cout << "action::centralizer_using_MAGMA before "
					"centralizer_using_magma2" << endl;
		}
		centralizer_using_magma2(prefix, fname_magma, fname_output,
				override_Sims, Elt, verbose_level);
		if (f_v) {
			cout << "action::centralizer_using_MAGMA after "
					"centralizer_using_magma2" << endl;
		}
	}
	if (f_v) {
		cout << "action::centralizer_using_MAGMA done" << endl;
	}
}

void action::read_centralizer_magma(std::string &fname_output,
		groups::sims *override_Sims, groups::strong_generators *&gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int number_of_generators;
	int *generator_perms;
	int goi, h, j;

	if (f_v) {
		cout << "action::read_centralizer_magma" << endl;
	}
	{
		ifstream fp(fname_output);

		fp >> goi;
		fp >> number_of_generators;
		generator_perms = NEW_int(number_of_generators * degree);
		for (h = 0; h < number_of_generators; h++) {
			for (j = 0; j < degree; j++) {
				fp >> generator_perms[h * degree + j];
			}
		}
		for (h = 0; h < number_of_generators * degree; h++) {
			generator_perms[h]--;
		}
	}

	data_structures_groups::vector_ge *nice_gens;


	gens = NEW_OBJECT(groups::strong_generators);

	gens->init_from_permutation_representation(this,
			override_Sims,
			generator_perms,
			number_of_generators, goi, nice_gens,
			verbose_level);

	if (f_v) {
		ring_theory::longinteger_object go1;

		cout << "action::read_centralizer_magma "
			"after gens->init_from_permutation_representation" << endl;
		cout << "centralizer order = " << goi
			<< " : " << endl;
		cout << "action::read_centralizer_magma created generators for a group" << endl;
		gens->print_generators(cout);
		gens->print_generators_as_permutations();
		gens->group_order(go1);
		cout << "action::read_centralizer_magma "
				"The group has order " << go1 << endl;
	}

	FREE_int(generator_perms);
	FREE_OBJECT(nice_gens);

	if (f_v) {
		cout << "action::read_centralizer_magma done" << endl;
	}
}

void action::centralizer_using_magma2(std::string &prefix,
		std::string &fname_magma,
		std::string &fname_output,
		groups::sims *override_Sims, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n;

	if (f_v) {
		cout << "action::centralizer_using_magma2" << endl;
	}
	orbiter_kernel_system::file_io Fio;
	groups::strong_generators *G_gen;

	G_gen = NEW_OBJECT(groups::strong_generators);
	G_gen->init_from_sims(override_Sims, 0 /* verbose_level */);

	n = degree;
	if (f_v) {
		cout << "action::centralizer_using_magma2 n = " << n << endl;
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
			<< " of size " << Fio.file_size(fname_magma) << endl;



	orbiter_kernel_system::magma_interface Magma;

	Magma.run_magma_file(fname_magma, verbose_level);
	if (Fio.file_size(fname_output) == 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << Magma.Orbiter_session->magma_path << "magma " << fname_magma << endl;
		exit(1);
	}


	cout << "command centralizer in MAGMA has finished" << endl;

	FREE_OBJECT(G_gen);

	if (f_v) {
		cout << "action::centralizer_using_magma2 done" << endl;
	}
}


void action::find_subgroups_using_MAGMA(std::string &prefix,
		groups::sims *override_Sims,
		int subgroup_order,
		int &nb_subgroups,
		groups::strong_generators *&H_gens, groups::strong_generators *&N_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_magma;
	string fname_output;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "action::find_subgroups_using_MAGMA" << endl;
	}
	fname_magma.assign(prefix);
	fname_magma.append("_subgroup.magma");

	fname_output.assign(prefix);
	fname_output.append("_subgroup.txt");


	if (Fio.file_size(fname_output) > 0) {
		read_subgroups_magma(fname_output, override_Sims, subgroup_order,
				nb_subgroups, H_gens, N_gens, verbose_level);
	}
	else {
		if (f_v) {
			cout << "action::find_subgroups_using_MAGMA before "
					"find_subgroups_using_MAGMA2" << endl;
		}
		find_subgroups_using_MAGMA2(prefix, fname_magma, fname_output,
				override_Sims, subgroup_order,
				verbose_level);
		if (f_v) {
			cout << "action::find_subgroups_using_MAGMA after "
					"find_subgroups_using_MAGMA2" << endl;
		}
		cout << "please run the magma file " << fname_magma
				<< ", retrieve the output file " << fname_output
				<< " and come back" << endl;
	}
	if (f_v) {
		cout << "action::find_subgroups_using_MAGMA done" << endl;
	}
}


void action::read_subgroups_magma(std::string &fname_output,
		groups::sims *override_Sims, int subgroup_order,
		int &nb_subgroups,
		groups::strong_generators *&H_gens, groups::strong_generators *&N_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	int u, h, j;

	if (f_v) {
		cout << "action::read_subgroups_magma" << endl;
	}

	{
		ifstream fp(fname_output);

		fp >> nb_subgroups;

		H_gens = NEW_OBJECTS(groups::strong_generators, nb_subgroups);
		N_gens = NEW_OBJECTS(groups::strong_generators, nb_subgroups);


		// read generators for H[]:

		for (u = 0; u < nb_subgroups; u++) {

			int *generator_perms;
			int number_of_generators;

			fp >> number_of_generators;
			generator_perms = NEW_int(number_of_generators * degree);
			for (h = 0; h < number_of_generators; h++) {
				for (j = 0; j < degree; j++) {
					fp >> generator_perms[h * degree + j];
				}
			}
			for (h = 0; h < number_of_generators * degree; h++) {
				generator_perms[h]--;
			}

			data_structures_groups::vector_ge *nice_gens;



			H_gens[u].init_from_permutation_representation(this,
					override_Sims,
					generator_perms,
					number_of_generators, subgroup_order, nice_gens,
					verbose_level);

			if (f_v) {
				ring_theory::longinteger_object go1;

				cout << "action::read_subgroups_magma "
					"after gens->init_from_permutation_representation" << endl;
				cout << "group order = " << subgroup_order
					<< " : " << endl;
				cout << "action::read_centralizer_magma created generators for a group" << endl;
				H_gens[u].print_generators(cout);
				H_gens[u].print_generators_as_permutations();
				H_gens[u].group_order(go1);
				cout << "action::read_subgroups_magma "
						"The group H[" << u << "] has order " << go1 << endl;
			}

			FREE_int(generator_perms);
			FREE_OBJECT(nice_gens);

		}

		// read generators for N[]:

		for (u = 0; u < nb_subgroups; u++) {

			int *generator_perms;
			int number_of_generators;
			int goi;

			fp >> goi;
			fp >> number_of_generators;
			generator_perms = NEW_int(number_of_generators * degree);
			for (h = 0; h < number_of_generators; h++) {
				for (j = 0; j < degree; j++) {
					fp >> generator_perms[h * degree + j];
				}
			}
			for (h = 0; h < number_of_generators * degree; h++) {
				generator_perms[h]--;
			}

			data_structures_groups::vector_ge *nice_gens;



			N_gens[u].init_from_permutation_representation(this,
					override_Sims,
					generator_perms,
					number_of_generators, goi, nice_gens,
					verbose_level);

			if (f_v) {
				ring_theory::longinteger_object go1;

				cout << "action::read_subgroups_magma "
					"after gens->init_from_permutation_representation" << endl;
				cout << "group order = " << subgroup_order
					<< " : " << endl;
				cout << "action::read_centralizer_magma created generators for a group" << endl;
				N_gens[u].print_generators(cout);
				N_gens[u].print_generators_as_permutations();
				N_gens[u].group_order(go1);
				cout << "action::read_subgroups_magma "
						"The group N[" << u << "] has order " << go1 << endl;
			}

			FREE_int(generator_perms);
			FREE_OBJECT(nice_gens);

		}

	}



	if (f_v) {
		cout << "action::read_subgroups_magma done" << endl;
	}
}

void action::find_subgroups_using_MAGMA2(std::string &prefix,
		std::string &fname_magma, std::string &fname_output,
		groups::sims *override_Sims, int subgroup_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	int n;

	if (f_v) {
		cout << "action::find_subgroups_using_MAGMA2" << endl;
	}


	string cmd;
	groups::strong_generators *G_gen;

	G_gen = NEW_OBJECT(groups::strong_generators);
	G_gen->init_from_sims(override_Sims, 0 /* verbose_level */);

	n = degree;
	if (f_v) {
		cout << "action::find_subgroups_using_MAGMA2 n = " << n << endl;
	}
	{
		ofstream fp(fname_magma);

		fp << "G := PermutationGroup< " << n << " | " << endl;
		G_gen->print_generators_MAGMA(this, fp);
		fp << ">;" << endl;



		fp << "Subgroups:=ElementaryAbelianSubgroups(G: OrderEqual:=" << subgroup_order << ");" << endl;
		fp << "Indicator:=[true: i in [1..#Subgroups]];" << endl;


		fp << "if #Subgroups ge 1 then" << endl;
		fp << "for i in [1..# Subgroups] do" << endl;
		fp << "H:= Subgroups[i]`subgroup;" << endl;
		fp << "f:=NumberingMap(H);" << endl;
		fp << "g:=Inverse(f);" << endl;
		fp << "for j in [1..Order(H)] do" << endl;
		fp << "if Order(Centralizer(G, G!g(j))) eq 368640 then" << endl;
		fp << "Indicator[i]:=false;" << endl;
		fp << "end if;" << endl;
		fp << "end for;" << endl;
		fp << "end for;" << endl;
		fp << "end if;" << endl;



		fp << "SetOutputFile(\"" << fname_output << "\");" << endl;

		fp << "S:= [j: j in [1..#Subgroups]|Indicator[j]];" << endl;
		fp << "#S;" << endl;
		//fp << "printf \"\\n\";" << endl;
		fp << "for i in S do" << endl;
		fp << "gens:=FewGenerators(Subgroups[i]`subgroup);" << endl;
		fp << "#gens;" << endl;
		fp << "for g in gens do" << endl;
		fp << "for k in [1..Degree(G)] do" << endl;
		fp << "    printf \"%o\", k^g; printf \" \";" << endl;
		fp << "  end for; " << endl;
		fp << "printf \"\\n\";" << endl;
		fp << "end for;" << endl;
		//fp << "printf \"\\n\";" << endl;
		fp << "end for;" << endl;


		//#S;
		//fp << "printf \"\\n\";" << endl;
		fp << "for i in S do" << endl;
		fp << "N:=Normalizer(G,Subgroups[i]`subgroup);" << endl;
		fp << "gens:=FewGenerators(N);" << endl;
		fp << "#N;" << endl;
		fp << "#gens;" << endl;
		fp << "for g in gens do" << endl;
		fp << "for k in [1..Degree(G)] do" << endl;
		fp << "	printf \"%o\", k^g; printf \" \";" << endl;
		fp << "  end for; " << endl;
		fp << "  printf \"\\n\";" << endl;
		fp << "end for;" << endl;
		//fp << "printf \"\\n\";" << endl;
		fp << "end for;" << endl;


		//fp << "printf \"%o\", #C; printf \"\\n\";" << endl;
		//fp << "printf \"%o\", #Generators(C); printf \"\\n\";" << endl;
		//fp << "for h := 1 to #Generators(C) do for i := 1 to "
		//		<< n << " do printf \"%o\", i^C.h; printf \" \"; end for;"
		//		" printf \"\\n\"; end for;" << endl;
		fp << "UnsetOutputFile();" << endl;
	}
	cout << "Written file " << fname_magma
			<< " of size " << Fio.file_size(fname_magma) << endl;

	orbiter_kernel_system::magma_interface Magma;

	Magma.run_magma_file(fname_magma, verbose_level);
	if (Fio.file_size(fname_output) == 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << Magma.Orbiter_session->magma_path << "magma " << fname_magma << endl;
		exit(1);
	}



	cout << "command script in MAGMA has finished" << endl;

	FREE_OBJECT(G_gen);



	if (f_v) {
		cout << "action::find_subgroups_using_MAGMA2 done" << endl;
	}
}

void action::conjugacy_classes_and_normalizers(groups::sims *override_Sims,
		std::string &label,
		std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string prefix;
	string fname_magma;
	string fname_output;
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers" << endl;
	}

	prefix.assign(label);
	fname_magma.assign(label);
	fname_magma.append("_classes.magma");
	fname_output.assign(label);
	fname_output.append("_classes_out.txt");


	if (Fio.file_size(fname_output) <= 0) {
		if (f_v) {
			cout << "action::conjugacy_classes_and_normalizers before "
					"conjugacy_classes_and_normalizers_using_MAGMA" << endl;
		}
		conjugacy_classes_and_normalizers_using_MAGMA(prefix,
				override_Sims, verbose_level);
		if (f_v) {
			cout << "action::conjugacy_classes_and_normalizers after "
					"conjugacy_classes_and_normalizers_using_MAGMA" << endl;
		}
	}


	if (Fio.file_size(fname_output) > 0) {
		if (f_v) {
			cout << "action::conjugacy_classes_and_normalizers before read_conjugacy_classes_and_normalizers" << endl;
		}
		read_conjugacy_classes_and_normalizers(fname_output, override_Sims, label_tex, verbose_level);
		if (f_v) {
			cout << "action::conjugacy_classes_and_normalizers after read_conjugacy_classes_and_normalizers" << endl;
		}
	}
	else {
		orbiter_kernel_system::magma_interface Magma;

		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << Magma.Orbiter_session->magma_path << "magma " << fname_magma << endl;
		exit(1);

	}

	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers done" << endl;
	}
}


void action::report_conjugacy_classes_and_normalizers(ostream &ost,
		groups::sims *override_Sims, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string prefix;
	string fname1;
	string fname2;
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers" << endl;
	}

	prefix.assign(label);
	fname1.assign(label);
	fname1.append("_classes.magma");
	fname2.assign(label);
	fname2.append("_classes_out.txt");


	if (Fio.file_size(fname2) > 0) {
		if (f_v) {
			cout << "action::conjugacy_classes_and_normalizers the file "
					<< fname2 << " exists, reading it " << endl;
		}
		read_and_report_conjugacy_classes_and_normalizers(ost,
				fname2, override_Sims, verbose_level);
	}
	else {
		if (f_v) {
			cout << "action::conjugacy_classes_and_normalizers the file " << fname2
					<< " does not exist, calling conjugacy_classes_and_normalizers_using_MAGMA" << endl;
		}
		if (!f_has_sims) {
			cout << "action::report_conjugacy_classes_and_normalizers we don't have sims, skipping" << endl;
		}
		else {
			conjugacy_classes_and_normalizers_using_MAGMA(prefix,
				Sims, verbose_level);
		}
	}

	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers done" << endl;
	}
}



void action::read_conjugacy_classes_and_normalizers(
		std::string &fname, groups::sims *override_sims,
		std::string &label_latex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_classes;
	int *perms;
	int *class_size;
	int *class_order_of_element;
	long int *class_normalizer_order;
	int *class_normalizer_number_of_generators;
	int **normalizer_generators_perms;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "action::read_conjugacy_classes_and_normalizers" << endl;
	}

	if (f_v) {
		cout << "action::read_conjugacy_classes_and_normalizers "
				"before read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
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
	if (f_v) {
		cout << "action::read_conjugacy_classes_and_normalizers "
				"after read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
	}

	cout << "i : class_order_of_element : class_normalizer_order" << endl;
	for (i = 0; i < nb_classes; i++) {
		cout << i << " : " << class_order_of_element[i] << " : " << class_normalizer_order[i] << endl;
	}


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


	ring_theory::longinteger_object go;
	ring_theory::longinteger_domain D;

	override_sims->group_order(go);
	cout << "The group has order " << go << endl;

	string fname_latex;
	data_structures::string_tools ST;

	fname_latex.assign(fname);

	ST.replace_extension_with(fname_latex, ".tex");

	{
		ofstream fp(fname_latex);
		string title, author, extra_praeamble;
		orbiter_kernel_system::latex_interface L;

		title.assign("Conjugacy classes of ");
		title.append("$");
		title.append(label_latex);
		title.append("$");

		author.assign("computed by Orbiter and MAGMA");

		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, author /* const char *author */,
			FALSE /* f_toc */, FALSE /* f_landscape */, TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */, TRUE /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);
		//latex_head_easy(fp);

		fp << "\\section{Conjugacy classes in $" << label_latex << "$}" << endl;


		fp << "The group order is " << endl;
		fp << "$$" << endl;
		go.print_not_scientific(fp);
		fp << endl;
		fp << "$$" << endl;

		cout << "second time" << endl;

		cout << "i : class_order_of_element : class_normalizer_order" << endl;
		for (i = 0; i < nb_classes; i++) {
			cout << i << " : " << class_order_of_element[i] << " : " << class_normalizer_order[i] << endl;
		}



		cout << "The conjugacy classes are:" << endl;
		for (i = 0; i < nb_classes; i++) {

			groups::strong_generators *gens;
			ring_theory::longinteger_object go1, Class_size, centralizer_order;
			long int goi;
			data_structures_groups::vector_ge *nice_gens;
			long int ngo;
			int nb_perms;
			groups::strong_generators *N_gens;
			data_structures_groups::vector_ge *nice_gens_N;


			cout << "The conjugacy class " << i << " / " << nb_classes << " is:" << endl;

			goi = class_order_of_element[i];
			ngo = class_normalizer_order[i];



			cout << "goi=" << goi << endl;
			cout << "ngo=" << ngo << endl;


			gens = NEW_OBJECT(groups::strong_generators);


			if (f_v) {
				cout << "action::read_conjugacy_classes_and_normalizers computing H, "
					"before gens->init_from_permutation_representation" << endl;
			}
			gens->init_from_permutation_representation(this, override_sims,
				perms + i * degree,
				1, goi, nice_gens,
				verbose_level - 5);

			if (f_v) {
				cout << "action::read_conjugacy_classes_and_normalizers computing H, "
					"after gens->init_from_permutation_representation" << endl;
			}

			Class_size.create(class_size[i], __FILE__, __LINE__);

			D.integral_division_exact(go, Class_size, centralizer_order);





			nb_perms = class_normalizer_number_of_generators[i];

			//int *class_normalizer_order;
			//int *class_normalizer_number_of_generators;
			//int **normalizer_generators_perms;

			if (f_v) {
				cout << "action::read_conjugacy_classes_and_normalizers computing N, "
					"before gens->init_from_permutation_representation" << endl;
			}
			N_gens = NEW_OBJECT(groups::strong_generators);
			N_gens->init_from_permutation_representation(this, override_sims,
					normalizer_generators_perms[i],
					nb_perms, ngo, nice_gens_N,
					verbose_level - 5);
			if (f_v) {
				cout << "action::read_conjugacy_classes_and_normalizers computing N, "
					"after gens->init_from_permutation_representation" << endl;
			}

			cout << "class " << i << " / " << nb_classes
				<< " size = " << class_size[i]
				<< " order of element = " << class_order_of_element[i]
				<< " centralizer order = " << centralizer_order
				<< " normalizer order = " << ngo
				<< " : " << endl;
			cout << "packing::read_conjugacy_classes_and_normalizers created "
					"generators for a group" << endl;
			gens->print_generators(cout);
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


			cout << "latex output element: " << endl;

			if (class_order_of_element[i] > 1) {
				Elt = nice_gens->ith(0);
				fp << "Representing element is" << endl;

				string label;
				char str[1000];

				sprintf(str, "c_{%d} = ", i);
				label.assign(str);

				element_print_latex_with_extras(Elt, label, fp);

	#if 0

				fp << "$$" << endl;
				element_print_latex(Elt, fp);
				fp << "$$" << endl;
	#endif

				fp << "$";
				element_print_for_make_element(Elt, fp);
				fp << "$\\\\" << endl;



			}

			cout << "latex output normalizer: " << endl;


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
		L.foot(fp);
	}
	cout << "Written file " << fname_latex << " of size "
			<< Fio.file_size(fname_latex) << endl;


	string fname_csv;

	fname_csv.assign(fname);

	ST.replace_extension_with(fname_csv, ".csv");
	{
		ofstream fp(fname_csv);
		fp << "ROW,class_order_of_element,class_size" << endl;
		for (i = 0; i < nb_classes; i++) {
			fp << i << "," << class_order_of_element[i] << "," << class_size[i] << endl;
		}
		fp << "END" << endl;

	}


	FREE_int(perms);
	FREE_int(class_size);
	FREE_int(class_order_of_element);
	FREE_lint(class_normalizer_order);
	FREE_int(class_normalizer_number_of_generators);
	FREE_pint(normalizer_generators_perms);
	//FREE_OBJECT(PA);

	if (f_v) {
		cout << "action::read_conjugacy_classes_and_normalizers done" << endl;
		}
}

void action::read_and_report_conjugacy_classes_and_normalizers(ostream &ost,
		std::string &fname, groups::sims *override_Sims, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_classes;
	int *perms;
	int *class_size;
	int *class_order_of_element;
	long int *class_normalizer_order;
	int *class_normalizer_number_of_generators;
	int **normalizer_generators_perms;
	//projective_space_with_action *PA;

	if (f_v) {
		cout << "action::read_and_report_conjugacy_classes_and_normalizers" << endl;
	}

	if (f_v) {
		cout << "action::read_and_report_conjugacy_classes_and_normalizers "
				"before read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
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
	if (f_v) {
		cout << "action::read_and_report_conjugacy_classes_and_normalizers "
				"after read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
	}


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


	ring_theory::longinteger_object go;
	ring_theory::longinteger_domain D;

	group_order(go);
	cout << "The group has order " << go << endl;

	string fname_latex;
	data_structures::string_tools ST;

	fname_latex.assign(fname);

	ST.replace_extension_with(fname_latex, ".tex");


	ost << "\\section{Conjugacy classes in $" << label_tex << "$}" << endl;


	ost << "The group order is " << endl;
	ost << "$$" << endl;
	go.print_not_scientific(ost);
	ost << endl;
	ost << "$$" << endl;


	cout << "The conjugacy classes are:" << endl;
	for (i = 0; i < nb_classes; i++) {
		groups::strong_generators *gens;
		ring_theory::longinteger_object go1, Class_size, centralizer_order;
		int goi;
		data_structures_groups::vector_ge *nice_gens;


		goi = class_order_of_element[i];
		gens = NEW_OBJECT(groups::strong_generators);

		gens->init_from_permutation_representation(this, override_Sims,
			perms + i * degree,
			1, goi, nice_gens,
			verbose_level);

		if (f_v) {
			cout << "action::normalizer_using_MAGMA "
				"after gens->init_from_permutation_"
				"representation" << endl;
		}

		Class_size.create(class_size[i], __FILE__, __LINE__);

		D.integral_division_exact(go, Class_size, centralizer_order);



		long int ngo;
		int nb_perms;
		groups::strong_generators *N_gens;
		data_structures_groups::vector_ge *nice_gens_N;

		ngo = class_normalizer_order[i];
		nb_perms = class_normalizer_number_of_generators[i];

		//int *class_normalizer_order;
		//int *class_normalizer_number_of_generators;
		//int **normalizer_generators_perms;

		N_gens = NEW_OBJECT(groups::strong_generators);
		N_gens->init_from_permutation_representation(this, override_Sims,
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
		gens->print_generators(cout);
		gens->print_generators_as_permutations();
		gens->group_order(go1);
		cout << "packing::read_conjugacy_classes_and_normalizers "
				"The group has order " << go1 << endl;

		ost << "\\bigskip" << endl;
		ost << "\\subsection*{Class " << i << " / "
				<< nb_classes << "}" << endl;
		ost << "Order of element = " << class_order_of_element[i]
				<< "\\\\" << endl;
		ost << "Class size = " << class_size[i] << "\\\\" << endl;
		ost << "Centralizer order = " << centralizer_order
				<< "\\\\" << endl;
		ost << "Normalizer order = " << ngo
				<< "\\\\" << endl;

		int *Elt = NULL;


		if (class_order_of_element[i] > 1) {
			Elt = nice_gens->ith(0);
			ost << "Representing element is" << endl;
			ost << "$$" << endl;
			element_print_latex(Elt, ost);
			ost << "$$" << endl;
			ost << "$";
			element_print_for_make_element(Elt, ost);
			ost << "$\\\\" << endl;



		}
		ost << "The normalizer is generated by:\\\\" << endl;
		N_gens->print_generators_tex(ost);


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
		} // next i

	FREE_int(perms);
	FREE_int(class_size);
	FREE_int(class_order_of_element);
	FREE_lint(class_normalizer_order);
	FREE_int(class_normalizer_number_of_generators);
	FREE_pint(normalizer_generators_perms);
	//FREE_OBJECT(PA);

	if (f_v) {
		cout << "action::read_and_report_conjugacy_classes_and_normalizers done" << endl;
		}
}


void action::report_groups_and_normalizers(std::ostream &ost,
		int nb_subgroups,
		groups::strong_generators *H_gens, groups::strong_generators *N_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int u;
	ring_theory::longinteger_object go1, go2;

	if (f_v) {
		cout << "action::report_groups_and_normalizers" << endl;
	}

	for (u = 0; u < nb_subgroups; u++) {

		ost << "\\subsection*{Class " << u << " / " << nb_subgroups << "}" << endl;

		H_gens[u].group_order(go1);
		N_gens[u].group_order(go2);

		ost << "Group order = " << go1 << "\\\\" << endl;
		ost << "Normalizer order = " << go2 << "\\\\" << endl;

		ost << "Generators for $H$:\\\\" << endl;

		H_gens[u].print_generators_in_latex_individually(ost);
		H_gens[u].print_generators_as_permutations_tex(ost, this);

		ost << "\\bigskip" << endl;

		ost << "Generators for $N(H)$:\\\\" << endl;

		N_gens[u].print_generators_in_latex_individually(ost);
		N_gens[u].print_generators_as_permutations_tex(ost, this);

	}


	if (f_v) {
		cout << "action::report_groups_and_normalizers done" << endl;
	}
}

void action::report_fixed_objects(int *Elt,
		char *fname_latex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, j, cnt;
	//int v[4];
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "action::report_fixed_objects" << endl;
		}


	{
		ofstream fp(fname_latex);
		string title, author, extra_praeamble;
		orbiter_kernel_system::latex_interface L;

		title.assign("Fixed Objects");

		L.head(fp,
			FALSE /* f_book */, TRUE /* f_title */,
			title, author /* const char *author */,
			FALSE /* f_toc */, FALSE /* f_landscape */, TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */, TRUE /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);
		//latex_head_easy(fp);

		fp << "\\section{Fixed Objects}" << endl;



		fp << "The element" << endl;
		fp << "$$" << endl;
		element_print_latex(Elt, fp);
		fp << "$$" << endl;
		fp << "has the following fixed objects:\\\\" << endl;


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


		L.foot(fp);
	}
	cout << "Written file " << fname_latex << " of size "
			<< Fio.file_size(fname_latex) << endl;


	if (f_v) {
		cout << "action::report_fixed_objects done" << endl;
		}
}

void action::element_conjugate_bvab(int *Elt_A,
		int *Elt_B, int *Elt_C, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2;


	if (f_v) {
		cout << "action::element_conjugate_bvab" << endl;
		}
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	if (f_v) {
		cout << "action::element_conjugate_bvab A=" << endl;
		element_print_quick(Elt_A, cout);
		cout << "action::element_conjugate_bvab B=" << endl;
		element_print_quick(Elt_B, cout);
		}

	element_invert(Elt_B, Elt1, 0);
	element_mult(Elt1, Elt_A, Elt2, 0);
	element_mult(Elt2, Elt_B, Elt_C, 0);
	if (f_v) {
		cout << "action::element_conjugate_bvab C=B^-1 * A * B" << endl;
		element_print_quick(Elt_C, cout);
		}
	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "action::element_conjugate_bvab done" << endl;
		}
}

void action::element_conjugate_babv(int *Elt_A,
		int *Elt_B, int *Elt_C, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2;


	if (f_v) {
		cout << "action::element_conjugate_babv" << endl;
		}
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);

	element_invert(Elt_B, Elt1, 0);
	element_mult(Elt_B, Elt_A, Elt2, 0);
	element_mult(Elt2, Elt1, Elt_C, 0);

	FREE_int(Elt1);
	FREE_int(Elt2);
}

void action::element_commutator_abavbv(int *Elt_A,
		int *Elt_B, int *Elt_C, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *Elt1, *Elt2, *Elt3, *Elt4;


	if (f_v) {
		cout << "action::element_commutator_abavbv" << endl;
		}
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	Elt3 = NEW_int(elt_size_in_int);
	Elt4 = NEW_int(elt_size_in_int);

	element_invert(Elt_A, Elt1, 0);
	element_invert(Elt_B, Elt2, 0);
	element_mult(Elt_A, Elt_B, Elt3, 0);
	element_mult(Elt3, Elt1, Elt4, 0);
	element_mult(Elt4, Elt2, Elt_C, 0);

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
}

void action::compute_projectivity_subgroup(
		groups::strong_generators *&projectivity_gens,
		groups::strong_generators *Aut_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::compute_projectivity_subgroup" << endl;
	}

	if (is_semilinear_matrix_group()) {
		if (f_v) {
			cout << "action::compute_projectivity_subgroup "
					"computing projectivity subgroup" << endl;
		}

		projectivity_gens = NEW_OBJECT(groups::strong_generators);
		{
			groups::sims *S;

			if (f_v) {
				cout << "action::compute_projectivity_subgroup "
						"before Aut_gens->create_sims" << endl;
			}
			S = Aut_gens->create_sims(0 /*verbose_level */);
			if (f_v) {
				cout << "action::compute_projectivity_subgroup "
						"after Aut_gens->create_sims" << endl;
			}
			if (f_v) {
				cout << "action::compute_projectivity_subgroup "
						"before projectivity_group_gens->projectivity_subgroup" << endl;
			}
			projectivity_gens->projectivity_subgroup(S, verbose_level - 3);
			if (f_v) {
				cout << "action::compute_projectivity_subgroup "
						"after projectivity_group_gens->projectivity_subgroup" << endl;
			}
			FREE_OBJECT(S);
		}
		if (f_v) {
			cout << "action::compute_projectivity_subgroup "
					"computing projectivity subgroup done" << endl;
		}
	}
	else {
		projectivity_gens = NULL;
	}


	if (f_v) {
		cout << "action::compute_projectivity_subgroup done" << endl;
	}
}



}}}

