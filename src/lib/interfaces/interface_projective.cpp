/*
 * interface_projective.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace interfaces {


interface_projective::interface_projective()
{
	argc = 0;
	argv = NULL;

	f_cheat_sheet_PG = FALSE;
	n = 0;
	q = 0;

	f_canonical_form_PG = FALSE;

	f_input = FALSE;
	Data_input_stream = NULL;

	f_all_k_subsets = FALSE;
	k = 0;

	f_save_incma_in_and_out = FALSE;

	f_prefix = FALSE;
	prefix = "";

	f_save = FALSE;
	output_prefix = "";

	fixed_structure_order_list_sz = 0;
	//fixed_structure_order_list[1000];

	f_report = FALSE;

	f_max_TDO_depth = FALSE;
	max_TDO_depth = INT_MAX;

}


void interface_projective::print_help(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-cheat_sheet_PG") == 0) {
		cout << "-cheat_sheet_PG" << endl;
	}
	else if (strcmp(argv[i], "-canonical_form_PG") == 0) {
		cout << "-canonical_form_PG" << endl;
	}
}

int interface_projective::recognize_keyword(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-cheat_sheet_PG <int : n> < int : q >") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-canonical_form_PG <int : n> < int : q >") == 0) {
		return true;
	}
	return false;
}

void interface_projective::read_arguments(int argc, const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_projective::read_arguments" << endl;
	//return 0;

	interface_projective::argc = argc;
	interface_projective::argv = argv;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-cheat_sheet_PG") == 0) {
			f_cheat_sheet_PG = TRUE;
			n = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-cheat_sheet_PG " << n << " " <<  q << endl;
			i++;
		}
		else if (strcmp(argv[i], "-canonical_form_PG") == 0) {
			f_canonical_form_PG = TRUE;
			n = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-canonical_form_PG " << n << " " <<  q << endl;
			i++;
			i = read_canonical_form_arguments(argc, argv, i, verbose_level);

		}
	}
	cout << "interface_projective::read_arguments done" << endl;
}

int interface_projective::read_canonical_form_arguments(int argc, const char **argv, int i0, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "interface_projective::read_canonical_form_arguments" << endl;
	}
	for (i = i0; i < argc; i++) {


		if (strcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data_input_stream = NEW_OBJECT(data_input_stream);
			i += Data_input_stream->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-input" << endl;
		}


		else if (strcmp(argv[i], "-all_k_subsets") == 0) {
			f_all_k_subsets = TRUE;
			k = atoi(argv[++i]);
			cout << "-all_k_subsets " << k << endl;
		}
		else if (strcmp(argv[i], "-save_incma_in_and_out") == 0) {
			f_save_incma_in_and_out = TRUE;
			cout << "-save_incma_in_and_out" << endl;
		}
		else if (strcmp(argv[i], "-prefix") == 0) {
			f_prefix = TRUE;
			prefix = argv[++i];
			cout << "-prefix " << prefix << endl;
		}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			output_prefix = argv[++i];
			cout << "-save " << output_prefix << endl;
		}
		else if (strcmp(argv[i], "-fixed_structure_of_element_of_order") == 0) {
			fixed_structure_order_list[fixed_structure_order_list_sz] = atoi(argv[++i]);
			cout << "-fixed_structure_of_element_of_order "
					<< fixed_structure_order_list[fixed_structure_order_list_sz] << endl;
			fixed_structure_order_list_sz++;
		}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}
		else if (strcmp(argv[i], "-max_TDO_depth") == 0) {
			f_max_TDO_depth = TRUE;
			max_TDO_depth = atoi(argv[++i]);
			cout << "-max_TDO_depth " << max_TDO_depth << endl;
		}
		else if (strcmp(argv[i], "-canonical_form_PG_end") == 0) {
			cout << "-canonical_form_PG_end " << endl;
			i++;
			break;
		}
		else {
			cout << "-canonical_form_PG: unrecognized option " << argv[i] << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "interface_projective::read_canonical_form_arguments done" << endl;
	}
	return i;
}

void interface_projective::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::worker" << endl;
	}

	if (f_cheat_sheet_PG) {
		do_cheat_sheet_PG(n, q, verbose_level);
	}
	else if (f_canonical_form_PG) {
		do_canonical_form_PG(n, q, verbose_level);
	}
	if (f_v) {
		cout << "interface_projective::worker done" << endl;
	}
}


void interface_projective::do_cheat_sheet_PG(int n, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "interface_projective::do_cheat_sheet_PG" << endl;
	}
	finite_field *F;

	F = NEW_OBJECT(finite_field);

	F->init(q, 0);
	//F->init_override_polynomial(q, my_override_poly, 0);

	//F->cheat_sheet_PG(n, f_surface, verbose_level);

	//const char *override_poly;
	char fname[1000];
	char title[1000];
	char author[1000];
	//int f_with_group = FALSE;
	//int f_semilinear = FALSE;
	//int f_basis = TRUE;
	//int q = F->q;

	sprintf(fname, "PG_%d_%d.tex", n, q);
	sprintf(title, "Cheat Sheet PG($%d,%d$)", n, q);
	//sprintf(author, "");
	author[0] = 0;
	projective_space *P;

	P = NEW_OBJECT(projective_space);
	cout << "before P->init" << endl;
	P->init(n, F,
		TRUE /* f_init_incidence_structure */,
		verbose_level/*MINIMUM(2, verbose_level)*/);


	{
	ofstream f(fname);
	latex_interface L;

	L.head(f,
			FALSE /* f_book*/,
			TRUE /* f_title */,
			title, author,
			FALSE /* f_toc */,
			FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	P->report(f);

	if (n == 3) {
		surface_domain *S;

		S = NEW_OBJECT(surface_domain);
		S->init(F, verbose_level + 2);

		f << "\\clearpage" << endl << endl;
		f << "\\section{Surface}" << endl;
		f << "\\subsection{Steiner Trihedral Pairs}" << endl;
		S->latex_table_of_trihedral_pairs(f);

		f << "\\clearpage" << endl << endl;
		f << "\\subsection{Eckardt Points}" << endl;
		S->latex_table_of_Eckardt_points(f);

#if 1
		long int *Lines;

		cout << "creating S_{3,1}:" << endl;
		Lines = NEW_lint(27);
		S->create_special_double_six(Lines,
				3 /*a*/, 1 /*b*/, 0 /* verbose_level */);
		S->create_remaining_fifteen_lines(Lines,
				Lines + 12, 0 /* verbose_level */);
		P->Grass_lines->print_set(Lines, 27);

		FREE_lint(Lines);
#endif
		FREE_OBJECT(S);
		}


	L.foot(f);
	}
	file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	FREE_OBJECT(P);

	FREE_OBJECT(F);

	if (f_v) {
		cout << "interface_projective::do_cheat_sheet_PG done" << endl;
	}

}

void interface_projective::do_canonical_form_PG(int n, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "interface_projective::do_canonical_form_PG" << endl;
	}

	if (!f_input) {
		cout << "please use option -input ... -end" << endl;
		exit(1);
	}

	finite_field *F;

	F = NEW_OBJECT(finite_field);

	F->init(q, 0);
	//F->init_override_polynomial(q, poly, 0);

	int f_semilinear;
	number_theory_domain NT;


	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}


	projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_space_with_action);

	PA->init(F, n,
		f_semilinear,
		TRUE /*f_init_incidence_structure*/,
		0 /* verbose_level */);





	classify_bitvectors *CB;

	CB = NEW_OBJECT(classify_bitvectors);


	cout << "canonical_form.cpp before PA->classify_objects_using_nauty" << endl;
	PA->classify_objects_using_nauty(Data_input_stream,
		CB,
		f_save_incma_in_and_out, prefix,
		verbose_level - 1);
	cout << "canonical_form.cpp after PA->classify_objects_using_nauty" << endl;



	cout << "canonical_form.cpp We found " << CB->nb_types << " types" << endl;


	compute_and_print_ago_distribution_with_classes(cout,
			CB, verbose_level);


	cout << "canonical_form.cpp In the ordering of canonical forms, they are" << endl;
	CB->print_reps();
	cout << "We found " << CB->nb_types << " types:" << endl;
	for (i = 0; i < CB->nb_types; i++) {

		object_in_projective_space_with_action *OiPA;
		object_in_projective_space *OiP;

		cout << i << " / " << CB->nb_types << " is "
			<< CB->Type_rep[i] << " : " << CB->Type_mult[i] << " : ";
		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[i];
		OiP = OiPA->OiP;
		if (OiP->type != t_PAC) {
			OiP->print(cout);
		}

#if 0
		for (j = 0; j < rep_len; j++) {
			cout << (int) Type_data[i][j];
			if (j < rep_len - 1) {
				cout << ", ";
				}
			}
#endif
		cout << endl;
	}



	if (f_save) {
		cout << "Saving the classification with output prefix "
				<< output_prefix << endl;
		PA->save(output_prefix, CB, verbose_level);
		CB->save(output_prefix,
			OiPA_encode, OiPA_group_order,
			NULL /* void *global_data */,
			verbose_level);

#if 0
		void save(const char *prefix,
			void (*encode_function)(void *extra_data,
				int *&encoding, int &encoding_sz, void *global_data),
			void (*get_group_order_or_NULL)(void *extra_data,
				longinteger_object &go, void *global_data),
			void *global_data,
			int verbose_level);
#endif
	}




	if (f_report) {

		cout << "Producing a latex report:" << endl;

		char fname[1000];

		if (prefix == NULL) {
			cout << "please use option -prefix <prefix> to set the "
					"prefix for the tex file" << endl;
			exit(1);
			}
		sprintf(fname, "%s_classification.tex", prefix);


		PA->latex_report(fname,
				output_prefix,
				CB,
				f_save_incma_in_and_out,
				fixed_structure_order_list_sz,
				fixed_structure_order_list,
				max_TDO_depth,
				verbose_level);

	}// f_report


	if (f_v) {
		cout << "interface_projective::do_canonical_form_PG done" << endl;
	}
}


}}

