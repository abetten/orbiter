/*
 * poset_classification_export_source_code.cpp
 *
 *  Created on: Nov 10, 2019
 *      Author: anton
 */



#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {



void poset_classification::generate_source_code(
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	string my_prefix;
	string fname;
	char str[1000];
	int iso_type;
	long int *rep;
	int i, j;
	int nb_iso;
	long int *set;
	ring_theory::longinteger_object go;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "poset_classification::generate_source_code" << endl;
	}

	my_prefix.assign(Control->problem_label);
	snprintf(str, sizeof(str), "_level_%d", level);
	my_prefix.append(str);

	fname.assign(my_prefix);
	fname.append(".cpp");

	set = NEW_lint(level);
	nb_iso = nb_orbits_at_level(level);




	{
		ofstream fp(fname);

		fp << "static long int " << Control->problem_label << "_nb_reps = " << nb_iso << ";" << endl;
		fp << "static long int " << Control->problem_label << "_size = " << level << ";" << endl;
		fp << "static long int " << Control->problem_label << "_reps[] = {" << endl;
		for (iso_type = 0; iso_type < nb_iso; iso_type++) {
			get_set_by_level(level, iso_type, set);
			rep = set;
			fp << "\t";
			for (i = 0; i < level; i++) {
				fp << rep[i];
				fp << ", ";
			}
			fp << endl;
		}
		fp << "};" << endl;
		fp << "static const char *" << Control->problem_label << "_stab_order[] = {" << endl;
		for (iso_type = 0; iso_type < nb_iso; iso_type++) {
			//rep = The_surface[iso_type]->coeff;

			data_structures_groups::set_and_stabilizer *SaS;

			SaS = get_set_and_stabilizer(level, iso_type,
					0 /* verbose_level */);
			fp << "\t\"";

			SaS->target_go.print_not_scientific(fp);
			fp << "\"," << endl;

			FREE_OBJECT(SaS);
		}
		fp << "};" << endl;


		fp << "static int " << Control->problem_label << "_make_element_size = "
				<< Poset->A->make_element_size << ";" << endl;

		{
		int *stab_gens_first;
		int *stab_gens_len;
		int fst;

		stab_gens_first = NEW_int(nb_iso);
		stab_gens_len = NEW_int(nb_iso);
		fst = 0;
		fp << "static int " << Control->problem_label << "_stab_gens[] = {" << endl;
		for (iso_type = 0; iso_type < nb_iso; iso_type++) {


			data_structures_groups::set_and_stabilizer *SaS;


			SaS = get_set_and_stabilizer(level,
					iso_type, 0 /* verbose_level */);

			stab_gens_first[iso_type] = fst;
			stab_gens_len[iso_type] = SaS->Strong_gens->gens->len;
			fst += stab_gens_len[iso_type];


			for (j = 0; j < stab_gens_len[iso_type]; j++) {
				if (f_vv) {
					cout << "poset_classification::generate_source_code "
							"before extract_strong_generators_in_order "
							"poset_classification "
							<< j << " / " << stab_gens_len[iso_type] << endl;
				}
				fp << "\t";
				Poset->A->Group_element->element_print_for_make_element(
						SaS->Strong_gens->gens->ith(j), fp);
				fp << endl;
			}

			FREE_OBJECT(SaS);

		}
		fp << "};" << endl;


		fp << "static long int " << Control->problem_label << "_stab_gens_fst[] = { ";
		for (iso_type = 0; iso_type < nb_iso; iso_type++) {
			fp << stab_gens_first[iso_type];
			if (iso_type < nb_iso - 1) {
				fp << ", ";
			}
			if (((iso_type + 1) % 10) == 0) {
				fp << endl << "\t";
			}
		}
		fp << "};" << endl;

		fp << "static long int " << Control->problem_label << "_stab_gens_len[] = { ";
		for (iso_type = 0; iso_type < nb_iso; iso_type++) {
			fp << stab_gens_len[iso_type];
			if (iso_type < nb_iso - 1) {
				fp << ", ";
			}
			if (((iso_type + 1) % 10) == 0) {
				fp << endl << "\t";
			}
		}
		fp << "};" << endl;




		FREE_int(stab_gens_first);
		FREE_int(stab_gens_len);
		}
	}

	FREE_lint(set);

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "poset_classification::generate_source_code done" << endl;
	}
}

void poset_classification::generate_history(int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	string my_prefix;
	string fname;
	char str[1000];
	int iso_type;
	long int *rep;
	int i, j;
	int nb_iso;
	long int *set;
	int *Elt;
	ring_theory::longinteger_object go;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "poset_classification::generate_history" << endl;
	}

	my_prefix.assign(Control->problem_label);
	snprintf(str, sizeof(str), "_history_level_%d", level);
	my_prefix.append(str);

	fname.assign(my_prefix);
	fname.append(".cpp");



	set = NEW_lint(level);
	Elt = NEW_int(Poset->A->elt_size_in_int);



	{
		ofstream fp(fname);
		int lvl;

		fp << "static int " << problem_label << "_make_element_size = "
				<< Poset->A->make_element_size << ";" << endl;


		for (lvl = 0; lvl <= level; lvl++) {

			cout << "poset_classification::generate_history lvl = " << lvl << " / " << level << endl;
			nb_iso = nb_orbits_at_level(lvl);

			fp << "// level " << lvl << ":" << endl;
			fp << "static long int " << problem_label << "_lvl_" << lvl << "_nb_reps = " << nb_iso << ";" << endl;
			fp << "static long int " << problem_label << "_lvl_" << lvl << "_size = " << level << ";" << endl;
			fp << "static long int " << problem_label << "_lvl_" << lvl << "_reps[] = {" << endl;
			for (iso_type = 0; iso_type < nb_iso; iso_type++) {
				get_set_by_level(lvl, iso_type, set);
				rep = set;
				fp << "\t";
				for (i = 0; i < lvl; i++) {
					fp << rep[i];
					fp << ",";
				}
				fp << endl;
			}
			fp << "};" << endl;


			int f_progress = FALSE;
			long int L = 0;
			long int L100 = 0;

			if (nb_iso > ONE_MILLION) {
				f_progress = TRUE;
				L = nb_iso;
				L100 = L / 100 + 1;
			}
			fp << "static const char *" << problem_label << "_lvl_" << lvl << "_stab_order[] = {" << endl << "\t";
			for (iso_type = 0; iso_type < nb_iso; iso_type++) {
				//rep = The_surface[iso_type]->coeff;


				if (test_if_stabilizer_is_trivial(lvl, iso_type,
						0 /* verbose_level */)) {
					fp << "\"1\",";

				}
				else {
					data_structures_groups::set_and_stabilizer *SaS;

					SaS = get_set_and_stabilizer(lvl, iso_type,
							0 /* verbose_level */);


					fp << "\"";
					SaS->target_go.print_not_scientific(fp);
					fp << "\",";
					//fp << ",";

					FREE_OBJECT(SaS);
				}
				if (((iso_type + 1) % 10) == 0) {
					fp << endl << "\t";
				}

				if (f_progress) {
					if ((iso_type % L100) == 0) {
						cout << "poset_classification::generate_history "
								"first loop at " << iso_type / L100 << "%" << endl;
					}
				}


				}
			fp << "};" << endl;


			{
				int *stab_gens_first;
				int *stab_gens_len;
				int fst;

				stab_gens_first = NEW_int(nb_iso);
				stab_gens_len = NEW_int(nb_iso);
				fst = 0;
				fp << "static int " << problem_label << "_lvl_" << lvl << "_stab_gens[] = {" << endl;
				for (iso_type = 0; iso_type < nb_iso; iso_type++) {


					if (test_if_stabilizer_is_trivial(lvl, iso_type,
							0 /* verbose_level */)) {

						stab_gens_first[iso_type] = fst;
						stab_gens_len[iso_type] = 0;

					}
					else {
						data_structures_groups::set_and_stabilizer *SaS;


						SaS = get_set_and_stabilizer(lvl,
								iso_type, 0 /* verbose_level */);

						stab_gens_first[iso_type] = fst;
						stab_gens_len[iso_type] = SaS->Strong_gens->gens->len;
						fst += stab_gens_len[iso_type];


						for (j = 0; j < stab_gens_len[iso_type]; j++) {
							if (f_vv) {
								cout << "poset_classification::generate_source_code "
										"before extract_strong_generators_in_order "
										"poset_classification "
										<< j << " / " << stab_gens_len[iso_type] << endl;
							}
							fp << "\t";
							Poset->A->Group_element->element_print_for_make_element(
									SaS->Strong_gens->gens->ith(j), fp);
							fp << endl;
						}
						FREE_OBJECT(SaS);
					}

					if (f_progress) {
						if ((iso_type % L100) == 0) {
							cout << "poset_classification::generate_history "
									"second loop at " << iso_type / L100 << "%" << endl;
						}
					}
				}
				fp << "};" << endl;


				fp << "static long int " << problem_label << "_lvl_" << lvl << "_stab_gens_fst[] = { " << endl << "\t";
				for (iso_type = 0; iso_type < nb_iso; iso_type++) {
					fp << stab_gens_first[iso_type];
					if (iso_type < nb_iso - 1) {
						fp << ",";
					}
					if (((iso_type + 1) % 10) == 0) {
						fp << endl << "\t";
					}
				}
				fp << "};" << endl;

				fp << "static long int " << problem_label << "_lvl_" << lvl << "_stab_gens_len[] = { " << endl << "\t";
				for (iso_type = 0; iso_type < nb_iso; iso_type++) {
					fp << stab_gens_len[iso_type];
					if (iso_type < nb_iso - 1) {
						fp << ",";
					}
					if (((iso_type + 1) % 10) == 0) {
						fp << endl << "\t";
					}
				}
				fp << "};" << endl;




				FREE_int(stab_gens_first);
				FREE_int(stab_gens_len);
			}

			{
				int *flag_orbit_first;
				int *flag_orbit_nb;
				int total_number_flag_orbits = 0;
				//int fst;

				flag_orbit_first = NEW_int(nb_iso);
				flag_orbit_nb = NEW_int(nb_iso);
				//fst = 0;
				for (iso_type = 0; iso_type < nb_iso; iso_type++) {

					poset_orbit_node *N;

					N = get_node_ij(lvl, iso_type);
					//N = root + Poo->first_node_at_level(lvl) + iso_type;

					flag_orbit_nb[iso_type] = N->get_nb_of_extensions();
					total_number_flag_orbits += N->get_nb_of_extensions();
				}
				flag_orbit_first[0] = 0;
				for (iso_type = 1; iso_type < nb_iso; iso_type++) {
					flag_orbit_first[iso_type] =
							flag_orbit_first[iso_type - 1] +
							flag_orbit_nb[iso_type - 1];
				}
				fp << "static long int " << problem_label << "_lvl_" << lvl << "_total_number_flag_orbits = " << total_number_flag_orbits << endl;
				fp << "static long int " << problem_label << "_lvl_" << lvl << "_flag_orbit_fst[] = { " << endl << "\t";
				for (iso_type = 0; iso_type < nb_iso; iso_type++) {
					fp << flag_orbit_first[iso_type];
					if (iso_type < nb_iso - 1) {
						fp << ",";
					}
					if (((iso_type + 1) % 10) == 0) {
						fp << endl << "\t";
					}
				}
				fp << "};" << endl;
				fp << "static long int " << problem_label << "_lvl_" << lvl << "_flag_orbit_nb[] = { " << endl << "\t";
				for (iso_type = 0; iso_type < nb_iso; iso_type++) {
					fp << flag_orbit_nb[iso_type];
					if (iso_type < nb_iso - 1) {
						fp << ",";
					}
					if (((iso_type + 1) % 10) == 0) {
						fp << endl << "\t";
					}
				}
				fp << "};" << endl;

				int f;
				int po, so;
				data_structures::sorting Sorting;

				fp << "static int " << problem_label << "_lvl_" << lvl << "_flag_orbit_type[] = { " << endl << "\t";
				for (f = 0; f < total_number_flag_orbits; f++) {

					if (!Sorting.int_vec_search(flag_orbit_first, nb_iso, f, po)) {
						po--;
					}
					so = f - flag_orbit_first[po];
					cout << "flag orbit f=" << f << " po=" << po << " so=" << so << endl;

					poset_orbit_node *N;

					N = get_node_ij(lvl, po);
					//N = root + Poo->first_node_at_level(lvl) + po;

					if (so >= N->get_nb_of_extensions()) {
						cout << "so >= N->get_nb_of_extensions()" << endl;
						cout << "N->get_nb_of_extensions()=" << N->get_nb_of_extensions() << endl;
						exit(1);
					}

					fp << N->get_E(so)->get_type();

					if (f < total_number_flag_orbits - 1) {
						fp << ",";
					}
					if (((f + 1) % 10) == 0) {
						fp << endl << "\t";
					}
				}
				fp << "};" << endl;


				fp << "static long int " << problem_label << "_lvl_" << lvl << "_flag_orbit_pt[] = { " << endl << "\t";
				for (f = 0; f < total_number_flag_orbits; f++) {

					if (!Sorting.int_vec_search(flag_orbit_first, nb_iso, f, po)) {
						po--;
					}
					so = f - flag_orbit_first[po];
					cout << "flag orbit f=" << f << " po=" << po << " so=" << so << endl;

					poset_orbit_node *N;

					N = get_node_ij(lvl, po);
					//N = root + Poo->first_node_at_level(lvl) + po;

					fp << N->get_E(so)->get_pt();

					if (f < total_number_flag_orbits - 1) {
						fp << ",";
					}
					if (((f + 1) % 10) == 0) {
						fp << endl << "\t";
					}
				}
				fp << "};" << endl;

				fp << "static long int " << problem_label << "_lvl_" << lvl << "_flag_orbit_ol[] = { " << endl << "\t";
				for (f = 0; f < total_number_flag_orbits; f++) {

					if (!Sorting.int_vec_search(flag_orbit_first, nb_iso, f, po)) {
						po--;
					}
					so = f - flag_orbit_first[po];
					cout << "flag orbit f=" << f << " po=" << po << " so=" << so << endl;

					poset_orbit_node *N;

					N = get_node_ij(lvl, po);
					//N = root + Poo->first_node_at_level(lvl) + po;
					fp << N->get_E(so)->get_orbit_len();

					if (f < total_number_flag_orbits - 1) {
						fp << ",";
					}
					if (((f + 1) % 10) == 0) {
						fp << endl << "\t";
					}
				}
				fp << "};" << endl;

				int *fusion_idx;
				int nb_fuse = 0;

				fusion_idx = NEW_int(total_number_flag_orbits);

				for (f = 0; f < total_number_flag_orbits; f++) {

					if (!Sorting.int_vec_search(flag_orbit_first, nb_iso, f, po)) {
						po--;
					}
					so = f - flag_orbit_first[po];
					cout << "flag orbit f=" << f << " po=" << po << " so=" << so << endl;

					poset_orbit_node *N;

					N = get_node_ij(lvl, po);
					//N = root + Poo->first_node_at_level(lvl) + po;
					if (N->get_E(so)->get_type() == 2) {
						fusion_idx[f] = nb_fuse;
						nb_fuse++;
					}
				}
				int nb_fuse_total;

				nb_fuse_total = nb_fuse;

				nb_fuse = 0;

				fp << "static long int " << problem_label << "_lvl_" << lvl << "_flag_index[] = { " << endl << "\t";
				for (f = 0; f < total_number_flag_orbits; f++) {

					if (!Sorting.int_vec_search(flag_orbit_first, nb_iso, f, po)) {
						po--;
					}
					so = f - flag_orbit_first[po];
					cout << "flag orbit f=" << f << " po=" << po << " so=" << so << endl;

					poset_orbit_node *N;

					N = get_node_ij(lvl, po);
					//N = root + Poo->first_node_at_level(lvl) + po;
					if (N->get_E(so)->get_type() == 1) {
						fp << N->get_E(so)->get_data();
					}
					else if (N->get_E(so)->get_type() == 2) {
						fp << nb_fuse;
						nb_fuse++;
					}

					if (f < total_number_flag_orbits - 1) {
						fp << ",";
					}
					if (((f + 1) % 10) == 0) {
						fp << endl << "\t";
					}
				}
				fp << "};" << endl;



				int *fuse_data;

				cout << "collecting data" << endl;
				fuse_data = NEW_int(nb_fuse_total * 3);
				nb_fuse = 0;
				for (f = 0; f < total_number_flag_orbits; f++) {

					if (!Sorting.int_vec_search(flag_orbit_first, nb_iso, f, po)) {
						po--;
					}
					so = f - flag_orbit_first[po];
					cout << "flag orbit f=" << f << " po=" << po << " so=" << so << endl;

					poset_orbit_node *N;

					N = get_node_ij(lvl, po);
					//N = root + Poo->first_node_at_level(lvl) + po;
					if (N->get_E(so)->get_type() == 2) {
						fuse_data[nb_fuse * 3 + 0] = N->get_E(so)->get_data();
						fuse_data[nb_fuse * 3 + 1] = N->get_E(so)->get_data1();
						fuse_data[nb_fuse * 3 + 2] = N->get_E(so)->get_data2();
						nb_fuse++;
					}
				}

				cout << "writing data1" << endl;
				fp << "static long int " << problem_label << "_lvl_" << lvl << "_flag_fuse_data1[] = { " << endl << "\t";
				for (nb_fuse = 0; nb_fuse < nb_fuse_total; nb_fuse++) {
					fp << fuse_data[nb_fuse * 3 + 1];
					if (nb_fuse < nb_fuse_total - 1) {
						fp << ",";
					}
					if (((nb_fuse + 1) % 10) == 0) {
						fp << endl << "\t";
					}
				}
				fp << "};" << endl;

				cout << "writing data2" << endl;
				fp << "static long int " << problem_label << "_lvl_" << lvl << "_flag_fuse_data2[] = { " << endl << "\t";
				for (nb_fuse = 0; nb_fuse < nb_fuse_total; nb_fuse++) {
					fp << fuse_data[nb_fuse * 3 + 2];
					if (nb_fuse < nb_fuse_total - 1) {
						fp << ",";
					}
					if (((nb_fuse + 1) % 10) == 0) {
						fp << endl << "\t";
					}
				}
				fp << "};" << endl;
				int hdl;

				cout << "writing iso" << endl;
				fp << "static long int " << problem_label << "_lvl_" << lvl << "_flag_fuse_iso[] = { " << endl;
				for (nb_fuse = 0; nb_fuse < nb_fuse_total; nb_fuse++) {
					hdl = fuse_data[nb_fuse * 3 + 0];
					Poset->A->Group_element->element_retrieve(hdl, Elt, 0);
					fp << "\t";
					Poset->A->Group_element->element_print_for_make_element(Elt, fp);
					fp << endl;
				}
				fp << "};" << endl;

				FREE_int(fuse_data);
				FREE_int(fusion_idx);
				FREE_int(flag_orbit_first);
				FREE_int(flag_orbit_nb);
			}

		} // next lvl
	} // end ofstream

	FREE_lint(set);
	FREE_int(Elt);

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "poset_classification::generate_history done" << endl;
	}
}


}}}

