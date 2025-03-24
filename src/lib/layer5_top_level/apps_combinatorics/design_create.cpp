/*
 * design_create.cpp
 *
 *  Created on: Sep 19, 2019
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {


design_create::design_create()
{
	Record_birth();
	Descr = NULL;

	Design_object = NULL;

#if 0
	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;
#endif

#if 0
	q = 0;
	F = NULL;
	k = 0;
#endif

	A_base = NULL;
	A = NULL;
	A2 = NULL;
	Aut = NULL;
	Aut_on_lines = NULL;

	degree = 0;

#if 0
	f_has_set = false;
	set = NULL;
	sz = 0;
#endif

	f_has_group = false;
	Sg = NULL;

#if 0
	f_has_block_partition = false;
	block_partition_class_size = 0;
#endif

	PA = NULL;
	P = NULL;

#if 0
	block = NULL;

	v = 0;
	b = 0;
	nb_inc = 0;
	f_has_incma = false;
	incma = NULL;
#endif

}

design_create::~design_create()
{
	Record_death();

	if (Design_object) {
		FREE_OBJECT(Design_object);
	}

#if 0
	if (F) {
		FREE_OBJECT(F);
	}
	if (set) {
		FREE_lint(set);
	}
#endif
	if (Sg) {
		FREE_OBJECT(Sg);
	}
	if (PA) {
		FREE_OBJECT(PA);
	}
#if 0
	if (block) {
		FREE_int(block);
	}
	if (incma) {
		FREE_int(incma);
	}
#endif
}

void design_create::init(
		apps_combinatorics::design_create_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "design_create::init" << endl;
	}
	design_create::Descr = Descr;


	Design_object = NEW_OBJECT(combinatorics::design_theory::design_object);


	if (Descr->f_field) {

		Design_object->F = Get_finite_field(Descr->field_label);
		Design_object->q = Design_object->F->q;

		if (f_v) {
			cout << "design_create::init q = " << Design_object->q << endl;
		}
	}

	if (Descr->f_family) {
		if (f_v) {
			cout << "design_create::init -family "
					"family_name=" << Descr->family_name << endl;
		}
		if (ST.stringcmp(Descr->family_name, "PG_2_q") == 0) {
			if (f_v) {
				cout << "design_create::init PG(2," << Design_object->q << ")" << endl;
			}
			if (!Descr->f_field) {
				cout << "please use option -field <label> to specify the field" << endl;
				exit(1);
			}
			create_design_PG_2_q(Design_object->F, Design_object->set, Design_object->sz, Design_object->k, verbose_level);

			Design_object->f_has_set = true;

			Design_object->v = Design_object->q * Design_object->q + Design_object->q + 1;
			Design_object->b = Design_object->v;


			Design_object->prefix = "PG_2_q" + std::to_string(Design_object->q);
			Design_object->label_txt = "PG_2_q" + std::to_string(Design_object->q);
			Design_object->label_tex = "PG\\_2\\_q" + std::to_string(Design_object->q);

		}
		else {
			cout << "design_create::init "
					"family name is not recognized" << endl;
			exit(1);
		}


		if (f_v) {
			cout << "design_create::init "
					"before compute_incidence_matrix_from_set_of_codes" << endl;
		}

		compute_incidence_matrix_from_set_of_codes(verbose_level);

		if (f_v) {
			cout << "design_create::init "
					"after compute_incidence_matrix_from_set_of_codes" << endl;
		}



	}
	else if (Descr->f_sporadic) {
		if (f_v) {
			cout << "design_create::init -sporadic "
					"sporadic_name=" << Descr->sporadic_name << endl;
		}
		if (ST.stringcmp(Descr->sporadic_name, "Baker_1978") == 0) {
			if (f_v) {
				cout << "design_create::init Baker's elliptic semiplane from 1978" << endl;
			}


			if (f_v) {
				cout << "design_create::init "
						"before Design_object->make_Baker_elliptic_semiplane_1978" << endl;
			}

			Design_object->make_Baker_elliptic_semiplane_1978(
					verbose_level - 1);

			if (f_v) {
				cout << "design_create::init "
						"after Design_object->make_Baker_elliptic_semiplane_1978" << endl;
			}


#if 0
			combinatorics::design_theory::design_theory_global Design_theory_global;


			if (f_v) {
				cout << "design_create::init "
						"before Design_theory_global.make_Baker_elliptic_semiplane_1978" << endl;
			}
			Design_theory_global.make_Baker_elliptic_semiplane_1978(incma, v, b, verbose_level - 1);
			if (f_v) {
				cout << "design_create::init "
						"after Design_theory_global.make_Baker_elliptic_semiplane_1978" << endl;
			}


			k = 7;
			block = NEW_int(k);
			nb_inc = k * b;

			f_has_set = false;

			f_has_incma = true;

#if 0
			int *block; // [k]

			int v;
			int b;
			int nb_inc;
			int f_has_incma;
			int *incma; // [v * b]
#endif

			prefix = "Baker1978";
			label_txt = "Baker1978";
			label_tex = "Baker1978";
#endif

		}
		else if (ST.stringcmp(Descr->sporadic_name, "Mathon_1987") == 0) {
			if (f_v) {
				cout << "design_create::init Mathon's elliptic semiplane from 1987" << endl;
			}

			if (f_v) {
				cout << "design_create::init "
						"before Design_object->make_Mathon_elliptic_semiplane_1987" << endl;
			}
			Design_object->make_Mathon_elliptic_semiplane_1987(
					verbose_level - 1);
			if (f_v) {
				cout << "design_create::init "
						"after Design_object->make_Mathon_elliptic_semiplane_1987" << endl;
			}

#if 0
			combinatorics::design_theory::design_theory_global Design_theory_global;


			if (f_v) {
				cout << "design_create::init "
						"before Design_theory_global.make_Mathon_elliptic_semiplane_1987" << endl;
			}
			Design_theory_global.make_Mathon_elliptic_semiplane_1987(incma, v, b, verbose_level - 1);
			if (f_v) {
				cout << "design_create::init "
						"after Design_theory_global.make_Mathon_elliptic_semiplane_1987" << endl;
			}


			k = 12;
			block = NEW_int(k);
			nb_inc = k * b;

			f_has_set = false;

			f_has_incma = true;

#if 0
			int *block; // [k]

			int v;
			int b;
			int nb_inc;
			int f_has_incma;
			int *incma; // [v * b]
#endif

			prefix = "Mathon1987";
			label_txt = "Mathon1987";
			label_tex = "Mathon1987";
#endif

		}


		else {
			cout << "design_create::init "
					"sporadic name is not recognized" << endl;
			exit(1);
		}



	}

	else if (Descr->f_from_incidence_matrix) {
		if (f_v) {
			cout << "design_create::init "
					"-from_incidence_matrix " << Descr->from_incidence_matrix_label << endl;
		}

		if (f_v) {
			cout << "design_create::init "
					"before Design_object->make_design_from_incidence_matrix" << endl;
		}
		Design_object->make_design_from_incidence_matrix(
				Descr->from_incidence_matrix_label,
				verbose_level - 1);
		if (f_v) {
			cout << "design_create::init "
					"after Design_object->make_design_from_incidence_matrix" << endl;
		}

#if 0
		combinatorics::design_theory::design_theory_global Design_theory_global;


		if (f_v) {
			cout << "design_create::init "
					"before Design_theory_global.make_design_from_incidence_matrix" << endl;
		}
		Design_theory_global.make_design_from_incidence_matrix(
				incma, v, b, k,
				Descr->from_incidence_matrix_label,
				verbose_level - 1);
		if (f_v) {
			cout << "design_create::init "
					"after Design_theory_global.make_design_from_incidence_matrix" << endl;
		}

		block = NEW_int(k);
		nb_inc = k * b;

		f_has_set = false;

		f_has_incma = true;

#if 0
		int *block; // [k]

		int v;
		int b;
		int nb_inc;
		int f_has_incma;
		int *incma; // [v * b]
#endif

		prefix = Descr->from_incidence_matrix_label;
		label_txt = Descr->from_incidence_matrix_label;
		label_tex = Descr->from_incidence_matrix_label;
#endif

	}

	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "design_create::init "
					"from catalogue not yet implemented" << endl;
		}
		//int nb_iso;
		//knowledge_base K;

		exit(1);

	}

	else if (Descr->f_list_of_base_blocks) {

		if (f_v) {
			cout << "design_create::init "
					"list of base blocks" << endl;
			cout << "design_create::init "
					"list_of_base_blocks_group_label=" << Descr->list_of_base_blocks_group_label << endl;
			cout << "design_create::init "
					"list_of_base_blocks_fname=" << Descr->list_of_base_blocks_fname << endl;
			cout << "design_create::init "
					"list_of_base_blocks_col=" << Descr->list_of_base_blocks_col << endl;
			cout << "design_create::init "
					"list_of_base_blocks_selection_fname=" << Descr->list_of_base_blocks_selection_fname << endl;
			cout << "design_create::init "
					"list_of_base_blocks_selection_col=" << Descr->list_of_base_blocks_selection_col << endl;
			cout << "design_create::init "
					"list_of_base_blocks_selection_idx=" << Descr->list_of_base_blocks_selection_idx << endl;
		}

		groups::any_group *AG;

		AG = Get_any_group(Descr->list_of_base_blocks_group_label);

		A_base = AG->A_base;
		A = AG->A;
		A2 = AG->A; // ToDo!


		other::orbiter_kernel_system::file_io Fio;
		other::data_structures::set_of_sets *SoS_base_blocks;

		Fio.Csv_file_support->read_column_as_set_of_sets(
				Descr->list_of_base_blocks_fname,
				Descr->list_of_base_blocks_col,
				SoS_base_blocks,
				0 /*verbose_level*/);
		if (f_v) {
			cout << "design_create::init "
					"total number of base blocks = " << SoS_base_blocks->nb_sets << endl;
		}


		std::string *Column;
		int len;

		Fio.Csv_file_support->read_column_of_strings(
				Descr->list_of_base_blocks_selection_fname,
				Descr->list_of_base_blocks_selection_col,
				Column, len,
				verbose_level);


		if (Descr->list_of_base_blocks_selection_idx >= len) {
			cout << "base block selection is out of range" << endl;
			exit(1);
		}


		other::data_structures::string_tools ST;

		int *Base_block_selection;
		int nb_base_blocks;

		string column_entry, column_entry_clean;

		column_entry = Column[Descr->list_of_base_blocks_selection_idx];

		if (f_v) {
			cout << "design_create::init "
					"column entry = " << column_entry << endl;
		}

		ST.drop_quotes(column_entry, column_entry_clean);

		Int_vec_scan(column_entry_clean, Base_block_selection, nb_base_blocks);

		if (f_v) {
			cout << "design_create::init "
					"Descr->list_of_base_blocks_selection_idx = " << Descr->list_of_base_blocks_selection_idx << endl;
			cout << "design_create::init "
					"nb_base_blocks = " << nb_base_blocks << endl;
			cout << "design_create::init "
					"base block selection base block selection is ";
			Int_vec_print(cout, Base_block_selection, nb_base_blocks);
			cout << endl;
		}


		long int *Blocks; // [b * k]


		apps_combinatorics::combinatorics_global Apps_Combi;


		if (f_v) {
			cout << "design_create::init "
					"before Apps_Combi.span_base_blocks" << endl;
		}
		Apps_Combi.span_base_blocks(
				AG,
				SoS_base_blocks,
				Base_block_selection, nb_base_blocks,
				Design_object->v, Design_object->b, Design_object->k,
				Blocks,
				verbose_level);
		if (f_v) {
			cout << "design_create::init "
					"after Apps_Combi.span_base_blocks" << endl;
		}




		//combinatorics::other_combinatorics::combinatorics_domain Combi;
		combinatorics::design_theory::design_theory_global Design;

		if (f_v) {
			cout << "design_create::init "
					"before Design.compute_incidence_matrix_from_blocks_lint" << endl;
		}
		Design.compute_incidence_matrix_from_blocks_lint(
				Design_object->v, Design_object->b, Design_object->k, Blocks,
				Design_object->incma, 0 /*verbose_level - 2*/);
		if (f_v) {
			cout << "design_create::init "
					"after Design.compute_incidence_matrix_from_blocks_lint" << endl;
		}


		Design_object->f_has_incma = true;
		if (false) {
			cout << "design_create::init incma:" << endl;
			Int_matrix_print(Design_object->incma, Design_object->v, Design_object->b);
		}


		FREE_lint(Blocks);
		FREE_int(Base_block_selection);
		FREE_OBJECT(SoS_base_blocks);
		delete [] Column;

		Design_object->prefix = "blocks_v" + std::to_string(Design_object->v)
						+ "_b" + std::to_string(Design_object->b)
						+ "_k" + std::to_string(Design_object->k);
		Design_object->label_txt = "blocks_v" + std::to_string(Design_object->v)
								+ "_b" + std::to_string(Design_object->b)
								+ "_k" + std::to_string(Design_object->k);
		Design_object->label_tex = "blocks\\_v" + std::to_string(Design_object->v)
								+ "\\_b" + std::to_string(Design_object->b)
								+ "\\_k" + std::to_string(Design_object->k);

	}


	else if (Descr->f_list_of_blocks_coded) {

		if (f_v) {
			cout << "design_create::init "
					"list of blocks_coded" << endl;
		}

		degree = Descr->list_of_blocks_coded_v;
		Design_object->k = Descr->list_of_blocks_coded_k;




		Get_lint_vector_from_label(
				Descr->list_of_blocks_coded_label, Design_object->set, Design_object->sz,
				0 /* verbose_level */);

		Design_object->f_has_set = true;
		Design_object->v = degree;
		Design_object->b = Design_object->sz;

		Design_object->prefix = "blocks_v" + std::to_string(degree) + "_k" + std::to_string(Design_object->k);
		Design_object->label_txt = "blocks_v" + std::to_string(degree) + "_k" + std::to_string(Design_object->k);
		Design_object->label_tex = "blocks\\_v" + std::to_string(degree) + "\\_k" + std::to_string(Design_object->k);


		A_base = NEW_OBJECT(actions::action);

		A_base->Known_groups->init_symmetric_group(degree, verbose_level);
		A = A_base; // ToDo copy object?

		//A2 = NEW_OBJECT(actions::action);
		A2 = A->Induced_action->induced_action_on_k_subsets(Design_object->k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = false;
		Sg = NULL;

		if (f_v) {
			cout << "design_create::init "
					"before compute_incidence_matrix_from_set_of_codes" << endl;
		}

		compute_incidence_matrix_from_set_of_codes(verbose_level);

		if (f_v) {
			cout << "design_create::init "
					"after compute_incidence_matrix_from_set_of_codes" << endl;
		}

	}
	else if (Descr->f_list_of_sets_coded) {

		if (f_v) {
			cout << "design_create::init "
					"list of sets" << endl;
		}

		degree = Descr->list_of_sets_coded_v;

		Get_lint_vector_from_label(
				Descr->list_of_sets_coded_label, Design_object->set, Design_object->sz,
				0 /* verbose_level */);
		//Lint_vec_scan(Descr->list_of_blocks_text, set, sz);

		Design_object->f_has_set = true;
		Design_object->v = degree;
		Design_object->b = Design_object->sz;


		Design_object->prefix = "sets_v" + std::to_string(degree);
		Design_object->label_txt = "sets_v" + std::to_string(degree);
		Design_object->label_tex = "sets\\_v" + std::to_string(degree);

		A_base = NEW_OBJECT(actions::action);

		A_base->Known_groups->init_symmetric_group(degree, verbose_level);
		A = A_base; // ToDo copy object?

		//A2 = NEW_OBJECT(actions::action);
		//A2->induced_action_on_k_subsets(*A, k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = false;
		Sg = NULL;


		if (f_v) {
			cout << "design_create::init "
					"before compute_incidence_matrix_from_set_of_codes" << endl;
		}

		compute_incidence_matrix_from_set_of_codes(verbose_level);

		if (f_v) {
			cout << "design_create::init "
					"after compute_incidence_matrix_from_set_of_codes" << endl;
		}


	}
	else if (Descr->f_list_of_blocks_coded_from_file) {

		if (f_v) {
			cout << "design_create::init "
					"list of blocks from file "
					<< Descr->list_of_blocks_coded_from_file_fname << endl;
		}

		degree = Descr->list_of_blocks_coded_v;
		Design_object->k = Descr->list_of_blocks_coded_k;

		other::orbiter_kernel_system::file_io Fio;
		int m, n;

		Fio.Csv_file_support->lint_matrix_read_csv(
				Descr->list_of_blocks_coded_from_file_fname,
				Design_object->set, m, n, verbose_level);


		if (n != 1) {
			cout << "design_create::init "
					"f_list_of_blocks_from_file n != 1" << endl;
			exit(1);
		}
		Design_object->sz = m;

		Design_object->f_has_set = true;
		Design_object->v = degree;
		Design_object->b = Design_object->sz;

		Design_object->prefix = "blocks_v" + std::to_string(degree) + "_k" + std::to_string(Design_object->k);
		Design_object->label_txt = "blocks_v" + std::to_string(degree) + "_k" + std::to_string(Design_object->k);
		Design_object->label_tex = "blocks\\_v" + std::to_string(degree) + "\\_k" + std::to_string(Design_object->k);


		A_base = NEW_OBJECT(actions::action);

		A_base->Known_groups->init_symmetric_group(degree, verbose_level);
		A = A_base; // ToDo copy object?

		//A2 = NEW_OBJECT(actions::action);
		A2 = A->Induced_action->induced_action_on_k_subsets(Design_object->k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = false;
		Sg = NULL;


		if (f_v) {
			cout << "design_create::init "
					"before compute_incidence_matrix_from_set_of_codes" << endl;
		}

		compute_incidence_matrix_from_set_of_codes(verbose_level);

		if (f_v) {
			cout << "design_create::init "
					"after compute_incidence_matrix_from_set_of_codes" << endl;
		}


	}


	else if (Descr->f_list_of_blocks_from_file) {

		if (f_v) {
			cout << "design_create::init "
					"f_list_of_blocks_from_file "
					<< Descr->list_of_blocks_from_file_fname
					<< " " << Descr->list_of_blocks_from_file_column
					<< endl;
		}

		degree = Descr->list_of_blocks_from_file_v;

		other::orbiter_kernel_system::file_io Fio;

		other::data_structures::set_of_sets *SoS;

		if (f_v) {
			cout << "design_create::init "
					"before Fio.Csv_file_support->read_column_as_set_of_sets" << endl;
		}
		Fio.Csv_file_support->read_column_as_set_of_sets(
				Descr->list_of_blocks_from_file_fname,
				Descr->list_of_blocks_from_file_column,
				SoS,
				verbose_level);
		if (f_v) {
			cout << "design_create::init "
					"after Fio.Csv_file_support->read_column_as_set_of_sets" << endl;
		}

		if (f_v) {
			cout << "design_create::init "
					"SoS=" << endl;
			SoS->print();
		}

		Design_object->v = degree;
		Design_object->b = SoS->nb_sets;


		// extend the incidence matrix by three blocks
		// in the beginning to distinguish the three parts:

		int h, u, i, s;

		Design_object->nb_inc = 0;
		Design_object->incma = NEW_int(Design_object->v * Design_object->b);
		Int_vec_zero(Design_object->incma, Design_object->v * Design_object->b);
		for (h = 0; h < Design_object->b; h++) {
			s = SoS->Set_size[h];
			for (u = 0; u < s; u++) {
				i = SoS->Sets[h][u];
				Design_object->incma[i * Design_object->b + h] = 1;
				Design_object->nb_inc++;
			}
		}

		Design_object->f_has_set = false;
		Design_object->f_has_incma = true;


		if (Descr->f_block_partition) {

			int class_size;

			class_size = Descr->block_partition_sz;

			if (f_v) {
				cout << "design_create::init "
						"block partition with classes of size " << class_size << endl;
			}

			Design_object->f_has_block_partition = true;
			if (Design_object->b % Descr->block_partition_sz) {
				cout << "design_create::init "
						"class size must divide the number of blocks" << endl;
				cout << "design_create::init "
						"number of blocks = " << Design_object->b << endl;
				cout << "design_create::init "
						"class size = " << class_size << endl;
				exit(1);
			}

			Design_object->block_partition_class_size = class_size;

		}



		Design_object->prefix = "blocks_v" + std::to_string(degree); // + "_k" + std::to_string(k);
		Design_object->label_txt = "blocks_v" + std::to_string(degree); // + "_k" + std::to_string(k);
		Design_object->label_tex = "blocks\\_v" + std::to_string(degree); // + "\\_k" + std::to_string(k);

		f_has_group = false;

		if (f_v) {
			cout << "design_create::init "
					"before compute_incidence_matrix_from_blocks" << endl;
		}

#if 0
		compute_incidence_matrix_from_blocks(
				blocks, b, k, 0 /*verbose_level*/);
#endif

		if (f_v) {
			cout << "design_create::init "
					"after compute_incidence_matrix_from_blocks" << endl;
		}


	}


	else if (Descr->f_wreath_product_designs) {

		if (f_v) {
			cout << "design_create::init "
					"f_wreath_product_designs" << endl;
		}

		int n;

		n = Descr->wreath_product_designs_n;
		Design_object->k = Descr->wreath_product_designs_k;


		//Orbiter->Lint_vec.scan(Descr->list_of_blocks_text, set, sz);

		degree = 2 * n;

		//combinatorics::other_combinatorics::combinatorics_domain Combi;
		combinatorics::design_theory::design_theory_global Design;
		long int nb_blocks;


		if (f_v) {
			cout << "design_create::init "
					"before Design.create_wreath_product_design" << endl;
		}
		Design.create_wreath_product_design(
				n, Design_object->k,
				Design_object->set, nb_blocks, verbose_level);
		if (f_v) {
			cout << "design_create::init "
					"after Design.create_wreath_product_design" << endl;
		}

		if (f_v) {
			cout << "design_create::init "
					"f_wreath_product_designs nb_blocks=" << nb_blocks << endl;
		}

		Design_object->sz = nb_blocks;
		//Orbiter->Lint_vec.scan(Descr->list_of_blocks_text, set, sz);

		Design_object->f_has_set = true;
		Design_object->v = degree;
		Design_object->b = Design_object->sz;

		Design_object->prefix = "wreath_product_designs_n" + std::to_string(n) + "_k" + std::to_string(Design_object->k);
		Design_object->label_txt = "wreath_product_designs_n" + std::to_string(n) + "_k" + std::to_string(Design_object->k);
		Design_object->label_tex = "wreath\\_product\\_designs\\_n" + std::to_string(n) + "\\_k" + std::to_string(Design_object->k);

		A_base = NEW_OBJECT(actions::action);

		A_base->Known_groups->init_symmetric_group(degree, verbose_level);
		A = A_base; // ToDo copy object?

		//A2 = NEW_OBJECT(actions::action);
		A2 = A->Induced_action->induced_action_on_k_subsets(Design_object->k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = false;
		Sg = NULL;


		if (f_v) {
			cout << "design_create::init "
					"before compute_incidence_matrix_from_set_of_codes" << endl;
		}

		compute_incidence_matrix_from_set_of_codes(verbose_level);

		if (f_v) {
			cout << "design_create::init "
					"after compute_incidence_matrix_from_set_of_codes" << endl;
		}



	}
	else if (Descr->f_linear_space_from_latin_square) {

		if (f_v) {
			cout << "design_create::init "
					"f_linear_space_from_latin_square" << endl;
		}

		int *Mtx;
		int m1, m2, s;



		Get_matrix(Descr->linear_space_from_latin_square_name, Mtx, m1, m2);

		if (m1 != m2) {
			cout << "design_create::init the matrix must be square" << endl;
			exit(1);
		}

		s = m1;



		degree = 3 * s;

		//combinatorics::other_combinatorics::combinatorics_domain Combi;
		combinatorics::design_theory::design_theory_global Design;
		long int nb_blocks;


		Design.create_linear_space_from_latin_square(
				Mtx, s,
				Design_object->v, Design_object->k,
				Design_object->set /* Blocks */, nb_blocks,
				verbose_level);

		if (f_v) {
			cout << "design_create::init "
					"f_linear_space_from_latin_square "
					"nb_blocks=" << nb_blocks << endl;
		}

		Design_object->sz = nb_blocks;

		Design_object->f_has_set = true;
		Design_object->v = degree;
		Design_object->b = Design_object->sz;

		Design_object->prefix = "latin_square_order" + std::to_string(s)
				+ "_" + Descr->linear_space_from_latin_square_name;
		Design_object->label_txt = "latin_square_order" + std::to_string(s)
				+ "_" + Descr->linear_space_from_latin_square_name;
		Design_object->label_tex = "latin\\_square\\_order" + std::to_string(s)
				+ "\\_" + Descr->linear_space_from_latin_square_name;

#if 0
		A = NEW_OBJECT(actions::action);

#if 0
		int f_no_base = false;

		if (Descr->f_no_group) {
			f_no_base = true;
		}
#endif

		A->Known_groups->init_symmetric_group(degree, verbose_level);

		//A2 = NEW_OBJECT(actions::action);
		A2 = A->Induced_action->induced_action_on_k_subsets(k, verbose_level);

		Aut = NULL;
		Aut_on_lines = NULL;
		f_has_group = false;
		Sg = NULL;
#endif
		f_has_group = false;


		if (f_v) {
			cout << "design_create::init "
					"before compute_incidence_matrix_from_set_of_codes" << endl;
		}

		compute_incidence_matrix_from_set_of_codes(verbose_level);

		if (f_v) {
			cout << "design_create::init "
					"after compute_incidence_matrix_from_set_of_codes" << endl;
		}

		// extend the incidence matrix by three blocks
		// in the beginning to distinguish the three parts:

		int *incma0;
		int b0, h, u, i, j;

		incma0 = Design_object->incma;
		b0 = Design_object->b;

		Design_object->b += 3;
		Design_object->incma = NEW_int(Design_object->v * Design_object->b);
		Int_vec_zero(Design_object->incma, Design_object->v * Design_object->b);
		for (h = 0; h < 3; h++) {
			for (u = 0; u < s; u++) {
				i = h * s + u;
				Design_object->incma[i * Design_object->b + h] = 1;
				Design_object->nb_inc++;
			}
		}
		for (j = 0; j < b0; j++) {
			for (i = 0; i < Design_object->v; i++) {
				Design_object->incma[i * Design_object->b + 3 + j] = incma0[i * b0 + j];
			}
		}
		FREE_int(incma0);
		Design_object->f_has_set = false;

		FREE_int(Mtx);


	}
	else if (Descr->f_orbit_on_sets) {

		if (f_v) {
			cout << "design_create::init "
					"f_orbit_on_sets" << endl;
		}

		Design_object->k = Descr->orbit_on_sets_size;
		int local_idx = Descr->orbit_on_sets_idx;

		//std::string orbit_on_sets_orbits;

		orbits::orbits_create *OC;

		OC = Get_orbits(Descr->orbit_on_sets_orbits);

		if (!OC->f_has_On_subsets) {
			cout << "design_create::init f_orbit_on_sets "
					"but orbit structure has no orbits on sets" << endl;
			exit(1);
		}

		poset_classification::poset_classification *PC;

		PC = OC->On_subsets;

		int *blocks;
		long int *Blocks;
		int orbit_length;

		if (f_v) {
			cout << "design_create::init "
					"before PC->get_whole_orbit" << endl;
		}

		PC->get_whole_orbit(
				Design_object->k, local_idx,
			Blocks, orbit_length, verbose_level - 2);

		if (f_v) {
			cout << "design_create::init "
					"after PC->get_whole_orbit" << endl;
		}

		if (f_v) {
			cout << "design_create::init "
					"blocks:" << endl;
			Lint_matrix_print(Blocks, orbit_length, Design_object->k);
		}

		blocks = NEW_int(orbit_length * Design_object->k);
		Lint_vec_copy_to_int(Blocks, blocks, orbit_length * Design_object->k);

		A_base = PC->get_poset()->A;
		A = PC->get_poset()->A2;

		if (f_v) {
			cout << "design_create::init "
					"A_base=" << endl;
			A_base->print_info();
			cout << "design_create::init "
					"A=" << endl;
			A->print_info();
		}

		Design_object->f_has_set = false;
		Design_object->v = A->degree;
		Design_object->b = orbit_length;


		Design_object->prefix = "blocks_v" + std::to_string(Design_object->v) + "_k" + std::to_string(Design_object->k);
		Design_object->label_txt = "blocks_v" + std::to_string(Design_object->v) + "_k" + std::to_string(Design_object->k);
		Design_object->label_tex = "blocks\\_v" + std::to_string(Design_object->v) + "\\_k" + std::to_string(Design_object->k);

		f_has_group = false;

		if (f_v) {
			cout << "design_create::init "
					"before Design_object->compute_incidence_matrix_from_blocks" << endl;
		}

		Design_object->compute_incidence_matrix_from_blocks(
				blocks, Design_object->b, Design_object->k,
				0 /*verbose_level*/);

		if (f_v) {
			cout << "design_create::init "
					"after Design_object->compute_incidence_matrix_from_blocks" << endl;
		}

		if (f_v) {
			cout << "design_create::init "
					"before creating a copy of the group generators" << endl;
		}

		f_has_group = true;

		//groups::strong_generators *Sg;

		Sg = OC->Group->Subgroup_gens->create_copy(verbose_level - 2);

		if (f_v) {
			cout << "design_create::init "
					"after creating a copy of the group generators" << endl;
		}

		if (f_v) {
			cout << "design_create::init "
					"Sg->A=" << endl;
			A->print_info();
		}

		FREE_int(blocks);
		FREE_lint(Blocks);


	}



	else {
		cout << "design_create::init "
				"no design created" << endl;
		Design_object->sz = 0;
		f_has_group = false;


		//exit(1);
	}


	if (Descr->f_label) {
		Design_object->label_txt = Descr->label_txt;
		Design_object->label_tex = Descr->label_tex;
	}

	if (f_has_group) {
		if (f_v) {
			cout << "design_create::init "
					"the stabilizer is:" << endl;
			Sg->print_generators_tex(cout);
		}
	}
	else {
		cout << "design_create::init "
				"stabilizer is not available" << endl;
	}



	if (f_v) {
		cout << "design_create::init done" << endl;
	}
}

void design_create::create_design_PG_2_q(
		algebra::field_theory::finite_field *F,
		long int *&set, int &sz, int &k,
		int verbose_level)
// creates a projective_space_with_action object
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::create_design_PG_2_q" << endl;
	}

	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;
	int j;
	int f_semilinear;
	//int *block;

	if (F->e > 1) {
		f_semilinear = true;
	}
	else {
		f_semilinear = false;
	}
	PA = NEW_OBJECT(projective_geometry::projective_space_with_action);
	PA->init(F, 2 /* n */, f_semilinear,
			true /*f_init_incidence_structure*/, verbose_level);

	P = PA->P;


	Design_object->k = Design_object->q + 1;
	k = Design_object->k;
	degree = P->Subspaces->N_points;

	Design_object->block = NEW_int(k);
	sz = P->Subspaces->N_lines;
	set = NEW_lint(sz);
	for (j = 0; j < sz; j++) {

		int h;

		for (h = 0; h < k; h++) {
			Design_object->block[h] = P->Subspaces->Implementation->lines(j, h);
		}

		Sorting.int_vec_heapsort(Design_object->block, k);

		set[j] = Combi.rank_k_subset(
				Design_object->block, P->Subspaces->N_points, k);
		if (f_v) {
			cout << "block " << j << " / " << sz << " : ";
			Int_vec_print(cout, Design_object->block, Design_object->k);
			cout << " : " << set[j] << endl;
		}
	}
	Sorting.lint_vec_heapsort(set, sz);
	if (f_v) {
		cout << "design : ";
		Lint_vec_print(cout, set, sz);
		cout << endl;
	}

	if (f_v) {
		cout << "design_create::create_design_PG_2_q "
				"creating actions" << endl;
	}
	A = NEW_OBJECT(actions::action);


	A->Known_groups->init_symmetric_group(degree, verbose_level);

	//A2 = NEW_OBJECT(actions::action);
	A2 = A->Induced_action->induced_action_on_k_subsets(k, verbose_level);

	Aut = PA->A;
	Aut_on_lines = PA->A_on_lines;
	f_has_group = true;
	Sg = Aut->Strong_gens;


	//Aonk = A2->G.on_k_subsets;

	if (f_v) {
		cout << "design_create::create_design_PG_2_q "
				"creating actions done" << endl;
	}

	if (f_v) {
		cout << "design_create::create_design_PG_2_q done" << endl;
	}
}

void design_create::unrank_block_in_PG_2_q(
		int *block,
		int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::unrank_block_in_PG_2_q rk=" << rk
				<< " P->N_points=" << P->Subspaces->N_points
				<< " k=" << Design_object->k << endl;
	}
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	Combi.unrank_k_subset(rk, block, P->Subspaces->N_points, Design_object->k);
	if (f_v) {
		cout << "design_create::unrank_block_in_PG_2_q block = ";
		Int_vec_print(cout, block, Design_object->k);
		cout << endl;
	}
	if (f_v) {
		cout << "design_create::unrank_block_in_PG_2_q done" << endl;
	}
}

int design_create::rank_block_in_PG_2_q(
		int *block,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk;

	if (f_v) {
		cout << "design_create::rank_block_in_PG_2_q" << endl;
	}
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;

	Sorting.int_vec_heapsort(block, Design_object->k);
	rk = Combi.rank_k_subset(block, P->Subspaces->N_points, Design_object->k);
	if (f_v) {
		cout << "design_create::rank_block_in_PG_2_q done" << endl;
	}
	return rk;
}

int design_create::get_nb_colors_as_two_design(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	int nb_c;

	if (f_v) {
		cout << "design_create::get_nb_colors_as_two_design" << endl;
	}
	nb_c = Combi.binomial2(P->Subspaces->N_points - 2);
	if (f_v) {
		cout << "design_create::get_nb_colors_as_two_design done" << endl;
	}
	return nb_c;
}

int design_create::get_color_as_two_design_assume_sorted(
		long int *design, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c, i;

	if (f_v) {
		cout << "design_create::get_color_as_two_design_assume_sorted" << endl;
	}
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	Combi.unrank_k_subset(
			design[0], Design_object->block, P->Subspaces->N_points, Design_object->k);
	if (Design_object->block[0] != 0) {
		cout << "block[0] != 0" << endl;
		exit(1);
	}
	if (Design_object->block[1] != 1) {
		cout << "block[1] != 1" << endl;
		exit(1);
	}
	for (i = 2; i < Design_object->k; i++) {
		Design_object->block[i] -= 2;
	}
	c = Combi.rank_k_subset(
			Design_object->block + 2, P->Subspaces->N_points - 2, Design_object->k - 2);
	if (f_v) {
		cout << "design_create::get_color_as_two_design_assume_sorted done" << endl;
	}
	return c;
}

void design_create::compute_incidence_matrix_from_set_of_codes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_set_of_codes" << endl;
	}

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_set_of_codes set = ";
		Lint_vec_print(cout, Design_object->set, Design_object->sz);
		cout << endl;
	}

	//combinatorics::other_combinatorics::combinatorics_domain Combi;
	combinatorics::design_theory::design_theory_global Design;


	if (Design_object->f_has_set) {

		if (Descr->f_list_of_sets_coded) {

			int h;

			Design.compute_incidence_matrix_from_sets(
					Design_object->v, Design_object->b, Design_object->set,
						Design_object->incma,
						verbose_level);

			Design_object->nb_inc = 0;
			for (h = 0; h < Design_object->v * Design_object->b; h++) {
				if (Design_object->incma[h]) {
					Design_object->nb_inc++;
				}
			}

		}
		else if (Descr->f_list_of_blocks_coded) {

			int h;
			int *Blocks;

			if (f_v) {
				cout << "design_create::compute_incidence_matrix_from_set_of_codes "
						"before Design.compute_blocks_from_coding" << endl;
			}
			Design.compute_blocks_from_coding(
					Design_object->v, Design_object->b,
					Descr->list_of_blocks_coded_k, Design_object->set,
					Blocks, verbose_level);
			if (f_v) {
				cout << "design_create::compute_incidence_matrix_from_set_of_codes "
						"after Design.compute_blocks_from_coding" << endl;
			}

			if (f_v) {
				cout << "design_create::compute_incidence_matrix_from_set_of_codes "
						"Blocks:" << endl;
				cout << "v = " << Design_object->v << endl;
				cout << "b = " << Design_object->b << endl;
				cout << "k = " << Descr->list_of_blocks_coded_k << endl;
				Int_matrix_print(Blocks, Design_object->b, Descr->list_of_blocks_coded_k);
			}

			if (f_v) {
				cout << "design_create::compute_incidence_matrix_from_set_of_codes "
						"before Design.compute_incidence_matrix_from_blocks" << endl;
			}
			Design.compute_incidence_matrix_from_blocks(
					Design_object->v, Design_object->b,
					Descr->list_of_blocks_coded_k, Blocks,
					Design_object->incma, verbose_level);
			if (f_v) {
				cout << "design_create::compute_incidence_matrix_from_set_of_codes "
						"after Design.compute_incidence_matrix_from_blocks" << endl;
			}

			Design_object->nb_inc = 0;
			for (h = 0; h < Design_object->v * Design_object->b; h++) {
				if (Design_object->incma[h]) {
					Design_object->nb_inc++;
				}
			}
			if (f_v) {
				cout << "design_create::compute_incidence_matrix_from_set_of_codes "
						"nb_inc = " << Design_object->nb_inc << endl;
			}

			FREE_int(Blocks);

		}
		else {

			Design_object->nb_inc = Design_object->k * Design_object->b;
			Design.compute_incidence_matrix(
					Design_object->v, Design_object->b, Design_object->k, Design_object->set,
					Design_object->incma, verbose_level);

		}
		Design_object->f_has_incma = true;
	}
	else {
		cout << "design_create::compute_incidence_matrix_from_set_of_codes "
				"please give a set" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_set_of_codes "
				"The incidence matrix is:" << endl;
		Int_matrix_print(Design_object->incma, Design_object->v, Design_object->b);
	}

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_set_of_codes done" << endl;
	}

}

#if 0
void design_create::compute_incidence_matrix_from_blocks(
		int *blocks, int nb_blocks, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_blocks" << endl;
	}

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_blocks blocks = " << endl;
		Int_matrix_print(blocks, nb_blocks, k);
		cout << endl;
	}

	b = nb_blocks;
	int i, j, h;

	incma = NEW_int(v * b);
	Int_vec_zero(incma, v * b);

	for (j = 0; j < nb_blocks; j++) {
		for (h = 0; h < k; h++) {
			i = blocks[j * k + h];
			incma[i * b + j] = 1;
		}
	}

	nb_inc = nb_blocks * k;

	f_has_incma = true;

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_blocks "
				"The incidence matrix is:" << endl;
		if (v + b > 50) {
			cout << "design_create::compute_incidence_matrix_from_blocks "
					"too large to print" << endl;
		}
		else {
			Int_matrix_print(incma, v, b);
		}
	}

	if (f_v) {
		cout << "design_create::compute_incidence_matrix_from_blocks done" << endl;
	}


}


void design_create::compute_blocks_from_incidence_matrix(
		long int *&blocks, int &nb_blocks, int &block_sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "design_create::compute_blocks_from_incidence_matrix" << endl;
	}

	nb_blocks = b;
	block_sz = k;
	int i, j, h;

	blocks = NEW_lint(b * block_sz);

	for (j = 0; j < b; j++) {
		h = 0;
		for (i = 0; i < v; i++) {
			if (incma[i * b + j]) {
				blocks[j * k + h++] = i;
			}
		}
		if (h != k) {
			cout << "the number of entries in the column "
					"which are one does not match" << endl;
			cout << "h = " << h << endl;
			cout << "k = " << k << endl;
			exit(1);
		}
	}

	if (f_v) {
		cout << "design_create::compute_blocks_from_incidence_matrix "
				"blocks = " << endl;
		Lint_matrix_print(blocks, nb_blocks, k);
		cout << endl;
	}



	if (f_v) {
		cout << "design_create::compute_blocks_from_incidence_matrix done" << endl;
	}

}
#endif



}}}



