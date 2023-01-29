/*
 * magma_interface.cpp
 *
 *  Created on: Dec 31, 2022
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace interfaces {



magma_interface::magma_interface()
{

}

magma_interface::~magma_interface()
{

}


void magma_interface::centralizer_of_element(
		actions::action *A, groups::sims *S,
		std::string &element_description,
		std::string &label, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	string prefix;

	if (f_v) {
		cout << "magma_interface::centralizer_of_element label=" << label
				<< " element_description=" << element_description << endl;
	}

	prefix.assign(A->label);
	prefix.append("_elt_");
	prefix.append(label);

	Elt = NEW_int(A->elt_size_in_int);

	int *data;
	int data_len;


	Int_vec_scan(element_description, data, data_len);


	if (data_len != A->make_element_size) {
		cout << "data_len != A->make_element_size" << endl;
		exit(1);
	}

	A->make_element(Elt, data, 0 /* verbose_level */);

	int o;

	o = A->element_order(Elt);
	if (f_v) {
		cout << "magma_interface::centralizer_of_element Elt:" << endl;
		A->element_print_quick(Elt, cout);
		cout << "magma_interface::centralizer_of_element on points:" << endl;
		A->element_print_as_permutation(Elt, cout);
		//cout << "algebra_global_with_action::centralizer_of_element on lines:" << endl;
		//A2->element_print_as_permutation(Elt, cout);
	}

	if (f_v) {
		cout << "magma_interface::centralizer_of_element "
				"the element has order " << o << endl;
	}



	if (f_v) {
		cout << "magma_interface::centralizer_of_element "
				"before centralizer_using_MAGMA" << endl;
	}

	groups::strong_generators *gens;

	centralizer_using_MAGMA(A, prefix,
			S, Elt, gens, verbose_level);


	if (f_v) {
		cout << "magma_interface::centralizer_of_element "
				"after centralizer_using_MAGMA" << endl;
	}


	if (f_v) {
		cout << "generators for the centralizer are:" << endl;
		gens->print_generators_tex();
	}



	{
		string fname, title, author, extra_praeamble;
		char str[1000];

		fname.assign(prefix);
		fname.append("_centralizer.tex");
		snprintf(str, 1000, "Centralizer of element %s", label.c_str());
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "magma_interface::centralizer_of_element "
						"before report" << endl;
			}
			gens->print_generators_tex(ost);

			if (f_v) {
				cout << "magma_interface::centralizer_of_element "
						"after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}


	FREE_int(data);

	if (f_v) {
		cout << "magma_interface::centralizer_of_element done" << endl;
	}
}


void magma_interface::normalizer_of_cyclic_subgroup(
		actions::action *A, groups::sims *S,
		std::string &element_description,
		std::string &label, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;
	string prefix;

	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_subgroup label=" << label
				<< " element_description=" << element_description << endl;
	}

	prefix.assign("normalizer_of_");
	prefix.append(label);
	prefix.append("_in_");
	prefix.append(A->label);

	Elt = NEW_int(A->elt_size_in_int);

	int *data;
	int data_len;


	Int_vec_scan(element_description, data, data_len);


	if (data_len != A->make_element_size) {
		cout << "data_len != A->make_element_size" << endl;
		exit(1);
	}
#if 0
	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_subgroup Matrix:" << endl;
		int_matrix_print(data, 4, 4);
	}
#endif

	A->make_element(Elt, data, 0 /* verbose_level */);

	int o;

	o = A->element_order(Elt);
	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_subgroup label=" << label
				<< " element order=" << o << endl;
	}

	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_subgroup Elt:" << endl;
		A->element_print_quick(Elt, cout);
		cout << endl;
		cout << "magma_interface::normalizer_of_cyclic_subgroup on points:" << endl;
		A->element_print_as_permutation(Elt, cout);
		//cout << "algebra_global_with_action::centralizer_of_element on lines:" << endl;
		//A2->element_print_as_permutation(Elt, cout);
	}

	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_subgroup "
				"the element has order " << o << endl;
	}



	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_subgroup "
				"before normalizer_of_cyclic_group_using_MAGMA" << endl;
	}

	groups::strong_generators *gens;

	normalizer_of_cyclic_group_using_MAGMA(A, prefix,
			S, Elt, gens, verbose_level);



	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_subgroup "
				"after normalizer_of_cyclic_group_using_MAGMA" << endl;
	}



	cout << "magma_interface::normalizer_of_cyclic_subgroup "
			"generators for the normalizer are:" << endl;
	gens->print_generators_tex();


	{

		string fname, title, author, extra_praeamble;
		char str[1000];

		fname.assign(prefix);
		fname.append(".tex");
		snprintf(str, 1000, "Normalizer of cyclic subgroup %s", label.c_str());
		title.assign(str);


		{
			ofstream ost(fname);
			orbiter_kernel_system::latex_interface L;

			L.head(ost,
					FALSE /* f_book*/,
					TRUE /* f_title */,
					title, author,
					FALSE /* f_toc */,
					FALSE /* f_landscape */,
					TRUE /* f_12pt */,
					TRUE /* f_enlarged_page */,
					TRUE /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);

			ring_theory::longinteger_object go;
			gens->group_order(go);
			ost << "The subgroup generated by " << endl;
			ost << "$$" << endl;
			A->element_print_latex(Elt, ost);
			ost << "$$" << endl;
			ost << "has order " << o << "\\\\" << endl;
			ost << "The normalizer has order " << go << "\\\\" << endl;
			if (f_v) {
				cout << "magma_interface::normalizer_of_cyclic_subgroup before report" << endl;
			}
			gens->print_generators_tex(ost);

			if (f_v) {
				cout << "magma_interface::normalizer_of_cyclic_subgroup after report" << endl;
			}


			L.foot(ost);

		}
		orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}





	FREE_int(data);
	FREE_int(Elt);
	FREE_OBJECT(gens);

	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_subgroup done" << endl;
	}
}


void magma_interface::find_subgroups(
		actions::action *A, groups::sims *S,
		int subgroup_order,
		std::string &label,
		int &nb_subgroups,
		groups::strong_generators *&H_gens,
		groups::strong_generators *&N_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string prefix;
	char str[1000];

	if (f_v) {
		cout << "magma_interface::find_subgroups label=" << label
				<< " subgroup_order=" << subgroup_order << endl;
	}
	prefix.assign(label);
	snprintf(str, sizeof(str), "_find_subgroup_of_order_%d", subgroup_order);
	prefix.append(str);



	if (f_v) {
		cout << "magma_interface::find_subgroups "
				"before find_subgroup_using_MAGMA" << endl;
	}


	find_subgroups_using_MAGMA(A, prefix,
			S, subgroup_order,
			nb_subgroups, H_gens, N_gens, verbose_level);


	if (f_v) {
		cout << "magma_interface::find_subgroups "
				"after find_subgroup_using_MAGMA" << endl;
	}


	//cout << "generators for the subgroup are:" << endl;
	//gens->print_generators_tex();


	if (f_v) {
		cout << "magma_interface::find_subgroups done" << endl;
	}
}


void magma_interface::print_generators_MAGMA(
		actions::action *A,
		groups::strong_generators *SG, std::ostream &ost)
{
	int i;

	for (i = 0; i < SG->gens->len; i++) {
		//cout << "Generator " << i << " / "
		// << gens->len << " is:" << endl;
		A->element_print_as_permutation_with_offset(
			SG->gens->ith(i), ost,
			1 /* offset */,
			TRUE /* f_do_it_anyway_even_for_big_degree */,
			FALSE /* f_print_cycles_of_length_one */,
			0 /* verbose_level */);
		if (i < SG->gens->len - 1) {
			ost << ", " << endl;
		}
	}
}

void magma_interface::export_group(actions::action *A,
		groups::strong_generators *SG, std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "magma_interface::export_group" << endl;
		A->print_info();
	}
	if (A->is_matrix_group()) {
		groups::matrix_group *M;
		int *Elt;
		int h, i, j;

		M = A->get_matrix_group();

#if 0
		if (M->f_semilinear) {
			cout << "cannot export to magma if semilinear" << endl;
			return;
		}
#endif
		field_theory::finite_field *F;

		F = M->GFq;
		if (F->e > 1) {
			int a;

			if (f_v) {
				cout << "magma_interface::export_group extension field" << endl;
			}
			ost << "F<w>:=GF(" << F->q << ");" << endl;
			ost << "G := GeneralLinearGroup(" << M->n << ", F);" << endl;
			ost << "H := sub< G | ";
			for (h = 0; h < SG->gens->len; h++) {
				Elt = SG->gens->ith(h);
				ost << "[";
				for (i = 0; i < M->n; i++) {
					for (j = 0; j < M->n; j++) {
						a = Elt[i * M->n + j];
						if (a < F->p) {
							ost << a;
						}
						else {
							ost << "w^" << F->log_alpha(a);
						}
						if (j < M->n - 1) {
							ost << ",";
						}
					}
					if (i < M->n - 1) {
						ost << ", ";
					}
				}
				ost << "]";
				if (h < SG->gens->len - 1) {
					ost << ", " << endl;
				}
			}
			ost << " >;" << endl;
			if (M->f_semilinear) {
				if (Elt[M->n * M->n]) {
					cout << "cannot export to magma if Frobenius is present." << endl;
					return;
				}
			}

		}
		else {
			ost << "G := GeneralLinearGroup(" << M->n << ", GF(" << F->q << "));" << endl;
			ost << "H := sub< G | ";
			for (h = 0; h < SG->gens->len; h++) {
				Elt = SG->gens->ith(h);
				ost << "[";
				for (i = 0; i < M->n; i++) {
					for (j = 0; j < M->n; j++) {
						ost << Elt[i * M->n + j];
						if (j < M->n - 1) {
							ost << ",";
						}
					}
					if (i < M->n - 1) {
						ost << ", ";
					}
				}
				ost << "]";
				if (h < SG->gens->len - 1) {
					ost << ", " << endl;
				}
			}
			ost << " >;" << endl;
		}
	}
	else {
		cout << "magma_interface::export_group not A->is_matrix_group()" << endl;

		actions::action_global AG;

		cout << "magma_interface::export_group type_g = ";
		AG.action_print_symmetry_group_type(cout,
				A->type_G);
		cout << endl;


		export_permutation_group_to_magma2(
				ost, A,
				SG, verbose_level);

		//exit(1);
	}
	if (f_v) {
		cout << "magma_interface::export_group done" << endl;
	}
}


//GL42 := GeneralLinearGroup(4, GF(2));
//> Ominus42 := sub< GL42 | [1,0,0,0, 1,1,0,1, 1,0,1,0, 0,0,0,1 ],
//>                               [0,1,0,0, 1,0,0,0, 0,0,1,0, 0,0,0,1 ],
//>                               [0,1,0,0, 1,0,0,0, 0,0,1,0, 0,0,1,1 ] >;

void magma_interface::export_permutation_group_to_magma(
		std::string &fname, actions::action *A2,
		groups::strong_generators *SG, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "magma_interface::export_permutation_group_to_magma" << endl;
	}
	{
		ofstream ost(fname);

		export_permutation_group_to_magma2(
				ost, A2,
				SG, verbose_level);

	}
	if (f_v) {
		cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "magma_interface::export_permutation_group_to_magma done" << endl;
	}
}

void magma_interface::export_permutation_group_to_magma2(
		std::ostream &ost,
		actions::action *A2,
		groups::strong_generators *SG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "magma_interface::export_permutation_group_to_magma2" << endl;
	}
	ost << "G := sub< Sym(" << A2->degree << ") |" << endl;
	for (i = 0; i < SG->gens->len; i++) {
		A2->element_print_as_permutation_with_offset(
			SG->gens->ith(i), ost,
			1 /* offset */,
			TRUE /* f_do_it_anyway_even_for_big_degree */,
			FALSE /* f_print_cycles_of_length_one */,
			0 /* verbose_level */);
		if (i < SG->gens->len - 1) {
			ost << ", " << endl;
		}
	}
	ost << ">;" << endl;

	if (f_v) {
		cout << "magma_interface::export_permutation_group_to_magma2 done" << endl;
	}
}

void magma_interface::export_group_to_magma_and_copy_to_latex(
		std::string &label_txt,
		std::ostream &ost,
		actions::action *A2,
		groups::strong_generators *SG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "magma_interface::export_group_to_magma_and_copy_to_latex" << endl;
	}
	string export_fname;

	export_fname.assign(label_txt);
	export_fname.append("_group.magma");

	export_permutation_group_to_magma(
			export_fname, A2, SG, verbose_level - 2);
	if (f_v) {
		cout << "written file " << export_fname << " of size "
				<< Fio.file_size(export_fname) << endl;
	}

	ost << "\\subsection*{Magma Export}" << endl;
	ost << "To export the group to Magma, "
			"use the following file\\\\" << endl;
	ost << "\\begin{verbatim}" << endl;

	{
		ifstream fp1(export_fname);
		char line[100000];

		while (TRUE) {
			if (fp1.eof()) {
				break;
			}

			//cout << "count_number_of_orbits_in_file reading
			//line, nb_sol = " << nb_sol << endl;
			fp1.getline(line, 100000, '\n');
			ost << line << endl;
		}

	}
	ost << "\\end{verbatim}" << endl;

	if (f_v) {
		cout << "magma_interface::export_group_to_magma_and_copy_to_latex done" << endl;
	}
}


void magma_interface::normalizer_using_MAGMA(
		actions::action *A,
		std::string &fname_magma_prefix,
		groups::sims *G, groups::sims *H,
		groups::strong_generators *&gens_N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_magma;
	string fname_output;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "magma_interface::normalizer_using_MAGMA" << endl;
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

	n = A->degree;
	if (f_v) {
		cout << "magma_interface::normalizer_using_MAGMA n = " << n << endl;
	}
	{
		ofstream fp(fname_magma);

		fp << "G := PermutationGroup< " << n << " | " << endl;
		G_gen->print_generators_MAGMA(A, fp);
		fp << ">;" << endl;


		fp << "H := sub< G |" << endl;
		H_gen->print_generators_MAGMA(A, fp);
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

		run_magma_file(fname_magma, verbose_level);
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << orbiter_kernel_system::Orbiter->magma_path << "magma " << fname_magma << endl;
		exit(1);
	}

	if (f_v) {
		cout << "file " << fname_output << " exists, reading it" << endl;
	}

	int i, j;
	int go, nb_gens;
	int *perms;

	if (f_v) {
		cout << "magma_interface::normalizer_using_MAGMA" << endl;
	}
	{
		ifstream fp(fname_output);

		fp >> go;
		fp >> nb_gens;
		if (f_v) {
			cout << "magma_interface::normalizer_using_MAGMA We found " << nb_gens
					<< " generators for a group of order " << go << endl;
		}

		perms = NEW_int(nb_gens * A->degree);

		for (i = 0; i < nb_gens; i++) {
			for (j = 0; j < A->degree; j++) {
				fp >> perms[i * A->degree + j];
			}
		}
		if (f_v) {
			cout << "magma_interface::normalizer_using_MAGMA we read all "
					"generators from file " << fname_output << endl;
		}
	}
	for (i = 0; i < nb_gens * A->degree; i++) {
		perms[i]--;
	}

	//longinteger_object go1;


	gens_N = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "magma_interface::normalizer_using_MAGMA "
			"before gens->init_from_permutation_"
			"representation" << endl;
	}

	data_structures_groups::vector_ge *nice_gens;

	gens_N->init_from_permutation_representation(A, G,
		perms,
		nb_gens, go, nice_gens,
		verbose_level);
	if (f_v) {
		cout << "magma_interface::normalizer_using_MAGMA "
			"after gens->init_from_permutation_"
			"representation" << endl;
	}
	FREE_OBJECT(nice_gens);

	cout << "magma_interface::normalizer_using_MAGMA "
		"after gens->init_from_permutation_representation gens_N=" << endl;
	gens_N->print_generators_for_make_element(cout);




	if (f_v) {
		cout << "magma_interface::normalizer_using_MAGMA done" << endl;
	}
}

void magma_interface::conjugacy_classes_using_MAGMA(
		actions::action *A,
		std::string &prefix,
		groups::sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_magma;
	string fname_output;

	if (f_v) {
		cout << "magma_interface::conjugacy_classes_using_MAGMA" << endl;
		}

	fname_magma.assign(prefix);
	fname_magma.append("conjugacy_classes.magma");
	fname_output.assign(prefix);
	fname_output.append("conjugacy_classes.txt");

	int n;

	groups::strong_generators *G_gen;

	G_gen = NEW_OBJECT(groups::strong_generators);
	G_gen->init_from_sims(G, 0 /* verbose_level */);

	n = A->degree;
	if (f_v) {
		cout << "magma_interface::conjugacy_classes_using_MAGMA n = " << n << endl;
		}
	{
	ofstream fp(fname_magma);

	fp << "G := PermutationGroup< " << n << " | " << endl;
	G_gen->print_generators_MAGMA(A, fp);
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



	orbiter_kernel_system::file_io Fio;

	run_magma_file(fname_magma, verbose_level);
	if (Fio.file_size(fname_output) == 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << orbiter_kernel_system::Orbiter->magma_path << "magma " << fname_magma << endl;
		exit(1);
	}


	FREE_OBJECT(G_gen);



	if (f_v) {
		cout << "magma_interface::conjugacy_classes_using_MAGMA done" << endl;
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

void magma_interface::conjugacy_classes_and_normalizers_using_MAGMA_make_fnames(
		actions::action *A,
		std::string &prefix,
		std::string &fname_magma,
		std::string &fname_output)
{
	fname_magma.assign(prefix);
	fname_magma.append("_classes.magma");
	fname_output.assign(prefix);
	fname_output.append("_classes_out.txt");
}

void magma_interface::conjugacy_classes_and_normalizers_using_MAGMA(
		actions::action *A,
		std::string &prefix,
		groups::sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_magma;
	string fname_output;

	if (f_v) {
		cout << "magma_interface::conjugacy_classes_and_normalizers_using_MAGMA" << endl;
	}
	conjugacy_classes_and_normalizers_using_MAGMA_make_fnames(A, prefix, fname_magma, fname_output);
	if (f_v) {
		cout << "magma_interface::conjugacy_classes_and_normalizers_using_MAGMA, fname_magma = " << fname_magma << endl;
		cout << "magma_interface::conjugacy_classes_and_normalizers_using_MAGMA, fname_output = " << fname_output << endl;
	}

	int n;

	groups::strong_generators *G_gen;

	G_gen = NEW_OBJECT(groups::strong_generators);
	if (f_v) {
		cout << "magma_interface::conjugacy_classes_and_normalizers_using_MAGMA "
				"before G_gen->init_from_sims" << endl;
	}
	G_gen->init_from_sims(G, verbose_level);
	if (f_v) {
		cout << "action::conjugacy_classes_and_normalizers_using_MAGMA "
				"after G_gen->init_from_sims" << endl;
	}

	n = A->degree;
	if (f_v) {
		cout << "magma_interface::conjugacy_classes_and_normalizers_using_MAGMA n = " << n << endl;
		cout << "magma_interface::conjugacy_classes_and_normalizers_using_MAGMA fname_magma = " << fname_magma << endl;
		cout << "magma_interface::conjugacy_classes_and_normalizers_using_MAGMA fname_output = " << fname_output << endl;
		}
	{
		ofstream fp(fname_magma);

		fp << "G := PermutationGroup< " << n << " | " << endl;
		G_gen->print_generators_MAGMA(A, fp);
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


	orbiter_kernel_system::file_io Fio;

	run_magma_file(fname_magma, verbose_level);
	if (Fio.file_size(fname_output) <= 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << orbiter_kernel_system::Orbiter->magma_path << "magma " << fname_magma << endl;
		exit(1);
	}


	if (f_v) {
		cout << "command ConjugacyClasses in MAGMA has finished" << endl;
	}

	FREE_OBJECT(G_gen);





	if (f_v) {
		cout << "magma_interface::conjugacy_classes_and_normalizers_using_MAGMA done" << endl;
		}
}


//> M24 := sub< Sym(24) |
//>  (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24),
//>  (2,16,9,6,8)(3,12,13,18,4)(7,17,10,11,22)(14,19,21,20,15),
//>  (1,22)(2,11)(3,15)(4,17)(5,9)(6,19)(7,13)(8,20)(10,16)(12,21)(14,18)(23,24)>;
//> M24;


void magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA(
		actions::action *A,
		std::string &fname,
		int &nb_classes,
		int *&perms,
		long int *&class_size,
		int *&class_order_of_element,
		long int *&class_normalizer_order,
		int *&class_normalizer_number_of_generators,
		int **&normalizer_generators_perms,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;

	if (f_v) {
		cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
		cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA "
				"fname=" << fname << endl;
		cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA "
				"degree=" << A->degree << endl;
		}
	{
		ifstream fp(fname);

		fp >> nb_classes;
		if (f_v) {
			cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA "
					"We found " << nb_classes
					<< " conjugacy classes" << endl;
		}

		perms = NEW_int(nb_classes * A->degree);
		class_size = NEW_lint(nb_classes);
		class_order_of_element = NEW_int(nb_classes);

		for (i = 0; i < nb_classes; i++) {
			fp >> class_order_of_element[i];
			if (f_v) {
				cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA "
						"class " << i << " / " << nb_classes
						<< " order=" << class_order_of_element[i] << endl;
			}
			fp >> class_size[i];
			if (f_v) {
				cout << "class_size[i] = " << class_size[i] << endl;
			}
			for (j = 0; j < A->degree; j++) {
				fp >> perms[i * A->degree + j];
			}
		}
		if (FALSE) {
			cout << "perms:" << endl;
			Int_matrix_print(perms, nb_classes, A->degree);
		}
		for (i = 0; i < nb_classes * A->degree; i++) {
			perms[i]--;
		}

		class_normalizer_order = NEW_lint(nb_classes);
		class_normalizer_number_of_generators = NEW_int(nb_classes);
		normalizer_generators_perms = NEW_pint(nb_classes);

		if (f_v) {
			cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA "
					"reading normalizer generators:" << endl;
		}
		for (i = 0; i < nb_classes; i++) {
			if (f_v) {
				cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA "
						"class " << i << " / " << nb_classes << endl;
			}
			fp >> class_normalizer_order[i];

			cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA "
					"class " << i << " class_normalizer_order[i]=" << class_normalizer_order[i] << endl;

			if (class_normalizer_order[i] <= 0) {
				cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA "
						"class_normalizer_order[i] <= 0" << endl;
				cout << "class_normalizer_order[i]=" << class_normalizer_order[i] << endl;
				exit(1);
			}
			if (f_v) {
				cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA "
						"class " << i << " / " << nb_classes
						<< " class_normalizer_order[i]=" << class_normalizer_order[i] << endl;
			}
			fp >> class_normalizer_number_of_generators[i];
			normalizer_generators_perms[i] =
					NEW_int(class_normalizer_number_of_generators[i] * A->degree);
			for (h = 0; h < class_normalizer_number_of_generators[i]; h++) {
				for (j = 0; j < A->degree; j++) {
					fp >> normalizer_generators_perms[i][h * A->degree + j];
				}
			}
			for (h = 0; h < class_normalizer_number_of_generators[i] * A->degree; h++) {
				normalizer_generators_perms[i][h]--;
			}
		}
		if (f_v) {
			cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA "
					"we read all class representatives "
					"from file " << fname << endl;
		}
	}
	if (f_v) {
		cout << "magma_interface::read_conjugacy_classes_and_normalizers_from_MAGMA done" << endl;
		}
}


void magma_interface::normalizer_of_cyclic_group_using_MAGMA(
		actions::action *A,
		std::string &fname_magma_prefix,
		groups::sims *G, int *Elt,
		groups::strong_generators *&gens_N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_group_using_MAGMA" << endl;
	}
	groups::sims *H;


	H = G->A->create_sims_from_single_generator_without_target_group_order(
			Elt, verbose_level);

#if 0
	H = NEW_OBJECT(groups::sims);
	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_group_using_MAGMA "
				"before H->init_cyclic_group_from_generator" << endl;
	}
	H->init_cyclic_group_from_generator(G->A, Elt, verbose_level);
	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_group_using_MAGMA "
				"after H->init_cyclic_group_from_generator" << endl;
	}
#endif

	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_group_using_MAGMA "
				"before normalizer_using_MAGMA" << endl;
	}
	normalizer_using_MAGMA(A,
		fname_magma_prefix,
		G, H, gens_N,
		verbose_level);
	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_group_using_MAGMA "
				"after normalizer_using_MAGMA" << endl;
	}

	if (f_v) {
		cout << "magma_interface::normalizer_of_cyclic_group_using_MAGMA done" << endl;
	}
}

void magma_interface::centralizer_using_MAGMA(
		actions::action *A,
		std::string &prefix,
		groups::sims *override_Sims, int *Elt,
		groups::strong_generators *&gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_magma;
	string fname_output;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "magma_interface::centralizer_using_MAGMA" << endl;
	}

	fname_magma.assign(prefix);
	fname_magma.append("_centralizer.magma");
	fname_output.assign(prefix);
	fname_output.append("_centralizer.txt");


	if (Fio.file_size(fname_output) > 0) {
		read_centralizer_magma(A, fname_output, override_Sims,
				gens, verbose_level);
	}
	else {
		if (f_v) {
			cout << "magma_interface::centralizer_using_MAGMA before "
					"centralizer_using_magma2" << endl;
		}
		centralizer_using_magma2(A, prefix, fname_magma, fname_output,
				override_Sims, Elt, verbose_level);
		if (f_v) {
			cout << "magma_interface::centralizer_using_MAGMA after "
					"centralizer_using_magma2" << endl;
		}
	}
	if (f_v) {
		cout << "magma_interface::centralizer_using_MAGMA done" << endl;
	}
}

void magma_interface::read_centralizer_magma(
		actions::action *A,
		std::string &fname_output,
		groups::sims *override_Sims,
		groups::strong_generators *&gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int number_of_generators;
	int *generator_perms;
	int goi, h, j;

	if (f_v) {
		cout << "magma_interface::read_centralizer_magma" << endl;
	}
	{
		ifstream fp(fname_output);

		fp >> goi;
		fp >> number_of_generators;
		generator_perms = NEW_int(number_of_generators * A->degree);
		for (h = 0; h < number_of_generators; h++) {
			for (j = 0; j < A->degree; j++) {
				fp >> generator_perms[h * A->degree + j];
			}
		}
		for (h = 0; h < number_of_generators * A->degree; h++) {
			generator_perms[h]--;
		}
	}

	data_structures_groups::vector_ge *nice_gens;


	gens = NEW_OBJECT(groups::strong_generators);

	gens->init_from_permutation_representation(A,
			override_Sims,
			generator_perms,
			number_of_generators, goi, nice_gens,
			verbose_level);

	if (f_v) {
		ring_theory::longinteger_object go1;

		cout << "magma_interface::read_centralizer_magma "
			"after gens->init_from_permutation_representation" << endl;
		cout << "centralizer order = " << goi
			<< " : " << endl;
		cout << "magma_interface::read_centralizer_magma created generators for a group" << endl;
		gens->print_generators(cout);
		gens->print_generators_as_permutations();
		gens->group_order(go1);
		cout << "magma_interface::read_centralizer_magma "
				"The group has order " << go1 << endl;
	}

	FREE_int(generator_perms);
	FREE_OBJECT(nice_gens);

	if (f_v) {
		cout << "magma_interface::read_centralizer_magma done" << endl;
	}
}

void magma_interface::centralizer_using_magma2(
		actions::action *A,
		std::string &prefix,
		std::string &fname_magma,
		std::string &fname_output,
		groups::sims *override_Sims, int *Elt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int n;

	if (f_v) {
		cout << "magma_interface::centralizer_using_magma2" << endl;
	}
	orbiter_kernel_system::file_io Fio;
	groups::strong_generators *G_gen;

	G_gen = NEW_OBJECT(groups::strong_generators);
	G_gen->init_from_sims(override_Sims, 0 /* verbose_level */);

	n = A->degree;
	if (f_v) {
		cout << "magma_interface::centralizer_using_magma2 n = " << n << endl;
	}
	{
		ofstream fp(fname_magma);

		fp << "G := PermutationGroup< " << n << " | " << endl;
		G_gen->print_generators_MAGMA(A, fp);
		fp << ">;" << endl;

		fp << "h := G ! ";
		A->element_print_as_permutation_with_offset(Elt, fp,
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



	run_magma_file(fname_magma, verbose_level);
	if (Fio.file_size(fname_output) == 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << orbiter_kernel_system::Orbiter->magma_path << "magma " << fname_magma << endl;
		exit(1);
	}


	cout << "command centralizer in MAGMA has finished" << endl;

	FREE_OBJECT(G_gen);

	if (f_v) {
		cout << "magma_interface::centralizer_using_magma2 done" << endl;
	}
}


void magma_interface::find_subgroups_using_MAGMA(
		actions::action *A,
		std::string &prefix,
		groups::sims *override_Sims,
		int subgroup_order,
		int &nb_subgroups,
		groups::strong_generators *&H_gens,
		groups::strong_generators *&N_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string fname_magma;
	string fname_output;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "magma_interface::find_subgroups_using_MAGMA" << endl;
	}
	fname_magma.assign(prefix);
	fname_magma.append("_subgroup.magma");

	fname_output.assign(prefix);
	fname_output.append("_subgroup.txt");


	if (Fio.file_size(fname_output) > 0) {
		read_subgroups_magma(A, fname_output, override_Sims, subgroup_order,
				nb_subgroups, H_gens, N_gens, verbose_level);
	}
	else {
		if (f_v) {
			cout << "magma_interface::find_subgroups_using_MAGMA before "
					"find_subgroups_using_MAGMA2" << endl;
		}
		find_subgroups_using_MAGMA2(A, prefix, fname_magma, fname_output,
				override_Sims, subgroup_order,
				verbose_level);
		if (f_v) {
			cout << "magma_interface::find_subgroups_using_MAGMA after "
					"find_subgroups_using_MAGMA2" << endl;
		}
		cout << "please run the magma file " << fname_magma
				<< ", retrieve the output file " << fname_output
				<< " and come back" << endl;
	}
	if (f_v) {
		cout << "magma_interface::find_subgroups_using_MAGMA done" << endl;
	}
}


void magma_interface::read_subgroups_magma(
		actions::action *A,
		std::string &fname_output,
		groups::sims *override_Sims, int subgroup_order,
		int &nb_subgroups,
		groups::strong_generators *&H_gens,
		groups::strong_generators *&N_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	int u, h, j;

	if (f_v) {
		cout << "magma_interface::read_subgroups_magma" << endl;
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
			generator_perms = NEW_int(number_of_generators * A->degree);
			for (h = 0; h < number_of_generators; h++) {
				for (j = 0; j < A->degree; j++) {
					fp >> generator_perms[h * A->degree + j];
				}
			}
			for (h = 0; h < number_of_generators * A->degree; h++) {
				generator_perms[h]--;
			}

			data_structures_groups::vector_ge *nice_gens;



			H_gens[u].init_from_permutation_representation(A,
					override_Sims,
					generator_perms,
					number_of_generators, subgroup_order, nice_gens,
					verbose_level);

			if (f_v) {
				ring_theory::longinteger_object go1;

				cout << "magma_interface::read_subgroups_magma "
					"after gens->init_from_permutation_representation" << endl;
				cout << "group order = " << subgroup_order
					<< " : " << endl;
				cout << "magma_interface::read_centralizer_magma "
						"created generators for a group" << endl;
				H_gens[u].print_generators(cout);
				H_gens[u].print_generators_as_permutations();
				H_gens[u].group_order(go1);
				cout << "magma_interface::read_subgroups_magma "
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
			generator_perms = NEW_int(number_of_generators * A->degree);
			for (h = 0; h < number_of_generators; h++) {
				for (j = 0; j < A->degree; j++) {
					fp >> generator_perms[h * A->degree + j];
				}
			}
			for (h = 0; h < number_of_generators * A->degree; h++) {
				generator_perms[h]--;
			}

			data_structures_groups::vector_ge *nice_gens;



			N_gens[u].init_from_permutation_representation(A,
					override_Sims,
					generator_perms,
					number_of_generators, goi, nice_gens,
					verbose_level);

			if (f_v) {
				ring_theory::longinteger_object go1;

				cout << "magma_interface::read_subgroups_magma "
					"after gens->init_from_permutation_representation" << endl;
				cout << "group order = " << subgroup_order
					<< " : " << endl;
				cout << "magma_interface::read_centralizer_magma "
						"created generators for a group" << endl;
				N_gens[u].print_generators(cout);
				N_gens[u].print_generators_as_permutations();
				N_gens[u].group_order(go1);
				cout << "magma_interface::read_subgroups_magma "
						"The group N[" << u << "] has order " << go1 << endl;
			}

			FREE_int(generator_perms);
			FREE_OBJECT(nice_gens);

		}

	}



	if (f_v) {
		cout << "magma_interface::read_subgroups_magma done" << endl;
	}
}

void magma_interface::find_subgroups_using_MAGMA2(
		actions::action *A,
		std::string &prefix,
		std::string &fname_magma, std::string &fname_output,
		groups::sims *override_Sims, int subgroup_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	int n;

	if (f_v) {
		cout << "magma_interface::find_subgroups_using_MAGMA2" << endl;
	}


	string cmd;
	groups::strong_generators *G_gen;

	G_gen = NEW_OBJECT(groups::strong_generators);
	G_gen->init_from_sims(override_Sims, 0 /* verbose_level */);

	n = A->degree;
	if (f_v) {
		cout << "magma_interface::find_subgroups_using_MAGMA2 n = " << n << endl;
	}
	{
		ofstream fp(fname_magma);

		fp << "G := PermutationGroup< " << n << " | " << endl;
		G_gen->print_generators_MAGMA(A, fp);
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


	run_magma_file(fname_magma, verbose_level);
	if (Fio.file_size(fname_output) == 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << orbiter_kernel_system::Orbiter->magma_path << "magma " << fname_magma << endl;
		exit(1);
	}



	cout << "command script in MAGMA has finished" << endl;

	FREE_OBJECT(G_gen);



	if (f_v) {
		cout << "magma_interface::find_subgroups_using_MAGMA2 done" << endl;
	}
}

void magma_interface::conjugacy_classes_and_normalizers(
		actions::action *A,
		groups::sims *override_Sims,
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
		cout << "magma_interface::conjugacy_classes_and_normalizers" << endl;
	}

	prefix.assign(label);
	fname_magma.assign(label);
	fname_magma.append("_classes.magma");
	fname_output.assign(label);
	fname_output.append("_classes_out.txt");


	if (Fio.file_size(fname_output) <= 0) {
		if (f_v) {
			cout << "magma_interface::conjugacy_classes_and_normalizers before "
					"conjugacy_classes_and_normalizers_using_MAGMA" << endl;
		}
		conjugacy_classes_and_normalizers_using_MAGMA(A, prefix,
				override_Sims, verbose_level);
		if (f_v) {
			cout << "magma_interface::conjugacy_classes_and_normalizers after "
					"conjugacy_classes_and_normalizers_using_MAGMA" << endl;
		}
	}


	if (Fio.file_size(fname_output) > 0) {
		if (f_v) {
			cout << "magma_interface::conjugacy_classes_and_normalizers "
					"before read_conjugacy_classes_and_normalizers" << endl;
		}
		read_conjugacy_classes_and_normalizers(A,
				fname_output, override_Sims, label_tex, verbose_level);
		if (f_v) {
			cout << "action::conjugacy_classes_and_normalizers "
					"after read_conjugacy_classes_and_normalizers" << endl;
		}
	}
	else {

		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << orbiter_kernel_system::Orbiter->magma_path << "magma " << fname_magma << endl;
		exit(1);

	}

	if (f_v) {
		cout << "magma_interface::conjugacy_classes_and_normalizers done" << endl;
	}
}


void magma_interface::report_conjugacy_classes_and_normalizers(
		actions::action *A,
		std::ostream &ost,
		groups::sims *override_Sims, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string prefix;
	string fname1;
	string fname2;
	orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "magma_interface::conjugacy_classes_and_normalizers" << endl;
	}

	prefix.assign(A->label);
	fname1.assign(A->label);
	fname1.append("_classes.magma");
	fname2.assign(A->label);
	fname2.append("_classes_out.txt");


	if (Fio.file_size(fname2) > 0) {
		if (f_v) {
			cout << "magma_interface::conjugacy_classes_and_normalizers the file "
					<< fname2 << " exists, reading it " << endl;
		}
		read_and_report_conjugacy_classes_and_normalizers(A, ost,
				fname2, override_Sims, verbose_level);
	}
	else {
		if (f_v) {
			cout << "magma_interface::conjugacy_classes_and_normalizers the file " << fname2
					<< " does not exist, calling conjugacy_classes_and_normalizers_using_MAGMA" << endl;
		}
		if (!A->f_has_sims) {
			cout << "magma_interface::report_conjugacy_classes_and_normalizers "
					"we don't have sims, skipping" << endl;
		}
		else {
			conjugacy_classes_and_normalizers_using_MAGMA(A, prefix,
					A->Sims, verbose_level);
		}
	}

	if (f_v) {
		cout << "magma_interface::conjugacy_classes_and_normalizers done" << endl;
	}
}



void magma_interface::read_conjugacy_classes_and_normalizers(
		actions::action *A,
		std::string &fname, groups::sims *override_sims,
		std::string &label_latex, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_classes;
	int *perms;
	long int *class_size;
	int *class_order_of_element;
	long int *class_normalizer_order;
	int *class_normalizer_number_of_generators;
	int **normalizer_generators_perms;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "magma_interface::read_conjugacy_classes_and_normalizers" << endl;
	}

	if (f_v) {
		cout << "magma_interface::read_conjugacy_classes_and_normalizers "
				"before read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
	}
	read_conjugacy_classes_and_normalizers_from_MAGMA(
			A,
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
		cout << "magma_interface::read_conjugacy_classes_and_normalizers "
				"after read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
	}

	cout << "i : class_order_of_element : class_normalizer_order" << endl;
	for (i = 0; i < nb_classes; i++) {
		cout << i << " : " << class_order_of_element[i] << " : " << class_normalizer_order[i] << endl;
	}



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
			cout << i << " : " << class_order_of_element[i]
				<< " : " << class_normalizer_order[i] << endl;
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
				cout << "magma_interface::read_conjugacy_classes_and_normalizers computing H, "
					"before gens->init_from_permutation_representation" << endl;
			}
			gens->init_from_permutation_representation(A, override_sims,
				perms + i * A->degree,
				1, goi, nice_gens,
				verbose_level - 5);

			if (f_v) {
				cout << "magma_interface::read_conjugacy_classes_and_normalizers computing H, "
					"after gens->init_from_permutation_representation" << endl;
			}

			Class_size.create(class_size[i], __FILE__, __LINE__);

			D.integral_division_exact(go, Class_size, centralizer_order);





			nb_perms = class_normalizer_number_of_generators[i];

			//int *class_normalizer_order;
			//int *class_normalizer_number_of_generators;
			//int **normalizer_generators_perms;

			if (f_v) {
				cout << "magma_interface::read_conjugacy_classes_and_normalizers computing N, "
					"before gens->init_from_permutation_representation" << endl;
			}
			N_gens = NEW_OBJECT(groups::strong_generators);
			N_gens->init_from_permutation_representation(A, override_sims,
					normalizer_generators_perms[i],
					nb_perms, ngo, nice_gens_N,
					verbose_level - 5);
			if (f_v) {
				cout << "magma_interface::read_conjugacy_classes_and_normalizers computing N, "
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

				snprintf(str, sizeof(str), "c_{%d} = ", i);
				label.assign(str);

				A->element_print_latex_with_extras(Elt, label, fp);

	#if 0

				fp << "$$" << endl;
				element_print_latex(Elt, fp);
				fp << "$$" << endl;
	#endif

				fp << "$";
				A->element_print_for_make_element(Elt, fp);
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
	FREE_lint(class_size);
	FREE_int(class_order_of_element);
	FREE_lint(class_normalizer_order);
	FREE_int(class_normalizer_number_of_generators);
	FREE_pint(normalizer_generators_perms);
	//FREE_OBJECT(PA);

	if (f_v) {
		cout << "magma_interface::read_conjugacy_classes_and_normalizers done" << endl;
		}
}

void magma_interface::read_and_report_conjugacy_classes_and_normalizers(
		actions::action *A,
		std::ostream &ost,
		std::string &fname, groups::sims *override_Sims,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int nb_classes;
	int *perms;
	long int *class_size;
	int *class_order_of_element;
	long int *class_normalizer_order;
	int *class_normalizer_number_of_generators;
	int **normalizer_generators_perms;

	if (f_v) {
		cout << "magma_interface::read_and_report_conjugacy_classes_and_normalizers" << endl;
	}

	if (f_v) {
		cout << "magma_interface::read_and_report_conjugacy_classes_and_normalizers "
				"before read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
	}
	read_conjugacy_classes_and_normalizers_from_MAGMA(
			A,
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
		cout << "magma_interface::read_and_report_conjugacy_classes_and_normalizers "
				"after read_conjugacy_classes_and_normalizers_from_MAGMA" << endl;
	}



	ring_theory::longinteger_object go;
	ring_theory::longinteger_domain D;

	A->group_order(go);
	cout << "The group has order " << go << endl;

	string fname_latex;
	data_structures::string_tools ST;

	fname_latex.assign(fname);

	ST.replace_extension_with(fname_latex, ".tex");


	ost << "\\section{Conjugacy classes in $" << A->label_tex << "$}" << endl;


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

		gens->init_from_permutation_representation(A, override_Sims,
			perms + i * A->degree,
			1, goi, nice_gens,
			verbose_level);

		if (f_v) {
			cout << "magma_interface::normalizer_using_MAGMA "
				"after gens->init_from_permutation_representation" << endl;
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
		N_gens->init_from_permutation_representation(A, override_Sims,
				normalizer_generators_perms[i],
				nb_perms, ngo, nice_gens_N,
				verbose_level - 1);

		cout << "class " << i << " / " << nb_classes
			<< " size = " << class_size[i]
			<< " order of element = " << class_order_of_element[i]
			<< " centralizer order = " << centralizer_order
			<< " normalizer order = " << ngo
			<< " : " << endl;
		cout << "magma_interface::read_conjugacy_classes_and_normalizers created "
				"generators for a group" << endl;
		gens->print_generators(cout);
		gens->print_generators_as_permutations();
		gens->group_order(go1);
		cout << "magma_interface::read_conjugacy_classes_and_normalizers "
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
			A->element_print_latex(Elt, ost);
			ost << "$$" << endl;
			ost << "$";
			A->element_print_for_make_element(Elt, ost);
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
	FREE_lint(class_size);
	FREE_int(class_order_of_element);
	FREE_lint(class_normalizer_order);
	FREE_int(class_normalizer_number_of_generators);
	FREE_pint(normalizer_generators_perms);
	//FREE_OBJECT(PA);

	if (f_v) {
		cout << "magma_interface::read_and_report_conjugacy_classes_and_normalizers done" << endl;
	}
}


void magma_interface::write_as_magma_permutation_group(
		groups::sims *S,
		std::string &fname_base,
		data_structures_groups::vector_ge *gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, k, n, l, h;
	ring_theory::longinteger_object go;
	int *Elt1;
	int *Elt2;
	int *Table;
	combinatorics::combinatorics_domain Combi;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "magma_interface::write_as_magma_permutation_group" << endl;
		}
	S->group_order(go);
	n = go.as_int();
	l = gens->len;
	if (f_v) {
		cout << "magma_interface::write_as_magma_permutation_group "
				"Computing the Table, go=" << go << endl;
		}
	Elt1 = NEW_int(S->A->elt_size_in_int);
	Elt2 = NEW_int(S->A->elt_size_in_int);
	Table = NEW_int(l * n);
	Int_vec_zero(Table, l * n);
	for (h = 0; h < l; h++) {
		if (f_v) {
			cout << "magma_interface::write_as_magma_permutation_group "
					"h = " << h <<  " / " << l << endl;
			}
		for (i = 0; i < n; i++) {
			if (f_v) {
				cout << "magma_interface::write_as_magma_permutation_group "
						"i = " << i <<  " / " << n << endl;
				}
			S->element_unrank_lint(i, Elt1);

			S->A->element_mult(Elt1, gens->ith(h), Elt2, 0);

			if (f_v) {
				cout << "Elt2=" << endl;
				S->A->element_print(Elt2, cout);
				}
			k = S->element_rank_lint(Elt2);
			if (f_v) {
				cout << "has rank k=" << k << endl;
				}
			Table[h * n + i] = k;
			}
		}
#if 0
> G := PermutationGroup< 12 | (1,6,7)(2,5,8,3,4,9)(11,12),
>                             (1,3)(4,9,12)(5,8,10,6,7,11) >;
#endif
	string fname;

	fname.assign(fname_base);
	fname.append(".magma");
	{
	ofstream fp(fname);

	fp << "G := PermutationGroup< " << n << " | " << endl;
	for (i = 0; i < l; i++) {
		Combi.perm_print_counting_from_one(fp, Table + i * n, n);
		if (i < l - 1) {
			fp << ", " << endl;
			}
		}
	fp << " >;" << endl;
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Table);

	if (f_v) {
		cout << "magma_interface::write_as_magma_permutation_group done" << endl;
		}
}

void magma_interface::export_linear_code(
		std::string &fname,
		field_theory::finite_field *F,
		int *genma, int n, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "magma_interface::export_linear_code" << endl;
	}

	ofstream ost(fname);
	int i, j, a;

	ost << "K<w> := GF(" << F->q << ");" << endl;
	ost << "V := VectorSpace(K, " << n << ");" << endl;
	ost << "C := LinearCode(sub<V |" << endl;
	for (i = 0; i < k; i++) {
		ost << "[";
		for (j = 0; j < n; j++) {
			a = genma[i * n + j];
			if (F->e == 1) {
				ost << a;
			}
			else {
				if (a <= 1) {
					ost << a;
				}
				else {
					ost << "w^" << F->log_alpha(a);
				}
			}
			if (j < n - 1) {
				ost << ",";
			}
		}
		ost << "]";
		if (i < k - 1) {
			ost << "," << endl;
		}
		else {
			ost << ">);" << endl;
		}
	}
	if (f_v) {
		cout << "magma_interface::export_linear_code done" << endl;
	}
}

void magma_interface::read_permutation_group(std::string &fname,
	int degree, int *&gens, int &nb_gens, int &go,
	int verbose_level)
{
	{
	ifstream fp(fname);
	int i, j, a;


	fp >> go;
	fp >> nb_gens;
	cout << "go = " << go << " nb_gens = " << nb_gens << endl;
	gens = NEW_int(nb_gens * degree);
	for (i = 0; i < nb_gens; i++) {
		for (j = 0; j < degree; j++) {
			fp >> a;
			a--;
			gens[i * degree + j] = a;
			}
		}
	}
}

void magma_interface::run_magma_file(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string cmd;

	cmd.assign(orbiter_kernel_system::Orbiter->magma_path);
	cmd.append("magma ");
	cmd.append(fname);

	if (f_v) {
		cout << "executing: " << cmd << endl;
	}
	system(cmd.c_str());
}

void magma_interface::normalizer_in_Sym_n(
		std::string &fname_base,
	int group_order, int *Table, int *gens, int nb_gens,
	int *&N_gens, int &N_nb_gens, int &N_go,
	int verbose_level)
{
	string fname_magma;
	string fname_output;
	int i;
	combinatorics::combinatorics_domain Combi;
	orbiter_kernel_system::file_io Fio;

	fname_magma.assign(fname_base);
	fname_magma.append(".magma");
	fname_output.assign(fname_base);
	fname_output.append(".txt");


	{
		ofstream fp(fname_magma);

		fp << "S := Sym(" << group_order << ");" << endl;


		fp << "G := PermutationGroup< " << group_order << " | " << endl;
		for (i = 0; i < nb_gens; i++) {
			Combi.perm_print_counting_from_one(fp,
				Table + gens[i] * group_order, group_order);
			if (i < nb_gens - 1) {
				fp << ", " << endl;
				}
			}
		fp << " >;" << endl;

		fp << "N := Normalizer(S, G);" << endl;
		fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
		fp << "printf \"%o\", #N; printf \"\\n\";" << endl;
		fp << "printf \"%o\", #Generators(N); printf \"\\n\";" << endl;
		fp << "for h := 1 to #Generators(N) do for i := 1 to "
			<< group_order << " do printf \"%o\", i^N.h; printf \" \"; "
			"if i mod 25 eq 0 then printf \"\n\"; end if; end for; "
			"printf \"\\n\"; end for;" << endl;
		fp << "UnsetOutputFile();" << endl;
	}


	if (Fio.file_size(fname_output) == 0) {

		run_magma_file(fname_magma, verbose_level);
		if (Fio.file_size(fname_output) == 0) {
			cout << "please run magma on the file " << fname_magma << endl;
			cout << "for instance, try" << endl;
			cout << orbiter_kernel_system::Orbiter->magma_path << "magma " << fname_magma << endl;
			exit(1);
		}
	}

	cout << "normalizer command in MAGMA has finished, written file "
		<< fname_output << " of size " << Fio.file_size(fname_output) << endl;


	read_permutation_group(fname_output,
			group_order, N_gens, N_nb_gens, N_go, verbose_level);

#if 0
	{
	ifstream fp(fname_output);


	fp >> N_go;
	fp >> N_nb_gens;
	cout << "N_go = " << N_go << " nb_gens = " << N_nb_gens << endl;
	N_gens = NEW_int(N_nb_gens * group_order);
	for (i = 0; i < N_nb_gens; i++) {
		for (j = 0; j < group_order; j++) {
			fp >> a;
			a--;
			N_gens[i * group_order + j] = a;
			}
		}
	}
#endif

}



#if 0
magma_interface::magma_interface()
{
	if (Orbiter == NULL) {
		cout << "magma_interface::magma_interface The_Orbiter_session == NULL" << endl;
		exit(1);
	}
	Orbiter_session = Orbiter;
}

magma_interface::~magma_interface()
{
}

void magma_interface::write_permutation_group(std::string &fname_base,
	int group_order, int *Table, int *gens, int nb_gens,
	int verbose_level)
{
	string fname;
	int i;
	combinatorics::combinatorics_domain Combi;
	file_io Fio;

	fname.assign(fname_base);
	fname.append(".magma");
	{
		ofstream fp(fname);

		fp << "G := PermutationGroup< " << group_order << " | " << endl;
		for (i = 0; i < nb_gens; i++) {
			Combi.perm_print_counting_from_one(fp,
					Table + gens[i] * group_order,
					group_order);
			if (i < nb_gens - 1) {
				fp << ", " << endl;
				}
			}
		fp << " >;" << endl;
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
}


void magma_interface::orbit_of_matrix_group_on_vector(
		std::string &fname_base,
	int d, int q,
	int *initial_vector, int **gens, int nb_gens,
	int &orbit_length,
	int verbose_level)
{
	string fname_magma;
	string fname_output;
	int i, j;
	combinatorics::combinatorics_domain Combi;
	file_io Fio;

	fname_magma.assign(fname_base);
	fname_magma.append(".magma");
	fname_output.assign(fname_base);
	fname_output.append(".txt");

	{
		ofstream fp(fname_magma);


		fp << "G := MatrixGroup< " << d << ", GF(" << q << ") | " << endl;
		for (i = 0; i < nb_gens; i++) {
			fp << "[";
			for (j = 0; j < d * d; j++) {
				fp << gens[i][j];
				if (j < d * d - 1) {
					fp << ",";
					}
			}
			fp << "]";
			if (i < nb_gens - 1) {
				fp << ", " << endl;
				}
			}
		fp << " >;" << endl;

		fp << "V := RSpace(G);" << endl;
		fp << "u := V![";
		for (j = 0; j < d; j++) {
			fp << initial_vector[j];
			if (j < d - 1) {
				fp << ",";
				}
		}

		fp << "];" << endl;
		fp << "O := Orbit(G,u);" << endl;


		fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
		fp << "printf \"%o\", #O; printf \"\\n\";" << endl;
		fp << "UnsetOutputFile();" << endl;
	}


	if (Fio.file_size(fname_output) == 0) {

		run_magma_file(fname_magma, verbose_level);
		if (Fio.file_size(fname_output) == 0) {
			cout << "please run magma on the file " << fname_magma << endl;
			cout << "for instance, try" << endl;
			cout << Orbiter_session->magma_path << "magma " << fname_magma << endl;
			exit(1);
		}
	}

	cout << "orbit command in MAGMA has finished, written file "
		<< fname_output << " of size " << Fio.file_size(fname_output) << endl;



	{
	ifstream fp(fname_output);


	fp >> orbit_length;
	}

}


void magma_interface::orbit_of_matrix_group_on_subspaces(
	std::string &fname_base,
	int d, int q, int k,
	int *initial_subspace, int **gens, int nb_gens,
	int &orbit_length,
	int verbose_level)
{
	string fname_magma;
	string fname_output;
	int i, j;
	combinatorics::combinatorics_domain Combi;
	file_io Fio;

	fname_magma.assign(fname_base);
	fname_magma.append(".magma");
	fname_output.assign(fname_base);
	fname_output.append(".txt");

	{
		ofstream fp(fname_magma);


		fp << "G := MatrixGroup< " << d << ", GF(" << q << ") | " << endl;
		for (i = 0; i < nb_gens; i++) {
			fp << "[";
			for (j = 0; j < d * d; j++) {
				fp << gens[i][j];
				if (j < d * d - 1) {
					fp << ",";
					}
			}
			fp << "]";
			if (i < nb_gens - 1) {
				fp << ", " << endl;
				}
			}
		fp << " >;" << endl;

		fp << "V := RSpace(G);" << endl;
		for (i = 0; i < k; i++) {
			fp << "u" << i << " := V![";
			for (j = 0; j < d; j++) {
				fp << initial_subspace[i * d + j];
				if (j < d - 1) {
					fp << ",";
					}
			}
			fp << "];" << endl;
		}

		fp << "W := sub< V | ";
		for (i = 0; i < k; i++) {
			fp << "u" << i;
			if (i < k - 1) {
				fp << ", ";
			}
		}
		fp << " >;" << endl;
		fp << "O := Orbit(G,W);" << endl;


		fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
		fp << "printf \"%o\", #O; printf \"\\n\";" << endl;
		fp << "UnsetOutputFile();" << endl;
	}


	if (Fio.file_size(fname_output) == 0) {
		run_magma_file(fname_magma, verbose_level);
		if (Fio.file_size(fname_output) == 0) {
			cout << "please run magma on the file " << fname_magma << endl;
			cout << "for instance, try" << endl;
			cout << Orbiter_session->magma_path << "magma " << fname_magma << endl;
			exit(1);
		}
	}

	cout << "orbit command in MAGMA has finished, written file "
		<< fname_output << " of size " << Fio.file_size(fname_output) << endl;



	{
	ifstream fp(fname_output);


	fp >> orbit_length;
	}

}

#endif


}}}


