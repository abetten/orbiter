/*
 * iso_type.cpp
 *
 *  Created on: Aug 15, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace geometry_builder {




iso_type::iso_type()
{
	Record_birth();
	gg = NULL;
	v = 0;
	sum_R_before = 0;
	sum_R = 0;

	f_orderly = false;

	f_generate_first = false;
	f_beginning_checked = false;

	f_split = false;
	split_remainder = 0;
	split_modulo = 1;



	//std::string fname;

	Canonical_forms = NULL;

	Nauty_control = NEW_OBJECT(other::l1_interfaces::nauty_interface_control);

	f_print_mod = true;
	print_mod = 1;

}

iso_type::~iso_type()
{
	Record_death();
}

void iso_type::init(
		gen_geo *gg,
		int v, int f_orderly, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::init v=" << v << endl;
	}

	if (v == 0) {
		cout << "iso_type::init v should not be zero" << endl;
		exit(1);
	}

	int i;

	iso_type::gg = gg;
	iso_type::v = v;
	iso_type::f_orderly = f_orderly;

	sum_R_before = 0;
	for (i = 0; i < v - 1; i++) {
		sum_R_before += gg->inc->Encoding->R[i];
	}
	sum_R = sum_R_before + gg->inc->Encoding->R[v - 1];

	if (f_v) {
		cout << "iso_type::init v=" << v
				<< " sum_R_before=" << sum_R_before
				<< " sum_R=" << sum_R
				<< endl;
	}

	Canonical_forms = NEW_OBJECT(combinatorics::canonical_form_classification::classify_using_canonical_forms);

	if (f_v) {
		cout << "iso_type::init done" << endl;
	}
}

void iso_type::add_geometry(
	inc_encoding *Encoding,
	int f_partition_fixing_last,
	int &f_already_there,
	int verbose_level)
{

	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "iso_type::add_geometry v=" << v << " verbose_level=" << verbose_level << endl;

		//inc->print(cout, v);
		//print_geometry(Encoding, v, inc);

	}

#if 0
	static long int count = 0;

	if (((1L << 18) - 1 & count) == 0) {
		cout << "iso_type::add_geometry v=" << v << " count = " << count << endl;

		//iso_type *it;
		int V = gg->GB->V;

		//it = gg->inc->iso_type_at_line[V - 1];

		//gg->print(cout, V, v);
		//cout << "iso_type::add_geometry after gg->print" << endl;
	}
	count++;
#endif

	int f_new_object;

	if (f_v) {
		cout << "iso_type::add_geometry v=" << v << " before find_and_add_geo" << endl;
	}
	find_and_add_geo(
		gg->inc->Encoding->theX,
		f_partition_fixing_last,
		f_new_object,
		verbose_level - 1);
	if (f_v) {
		cout << "iso_type::add_geometry v=" << v << " after find_and_add_geo" << endl;
	}

	if (f_new_object) {
		f_already_there = false;
	}
	else {
		f_already_there = true;
	}


	if (f_v) {
		cout << "iso_type::add_geometry done" << endl;
	}
}


void iso_type::find_and_add_geo(
	int *theY,
	int f_partition_fixing_last,
	int &f_new_object, int verbose_level)
{

	//verbose_level = 5;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout<< "iso_type::find_and_add_geo" << endl;
		gg->inc->print_geo(cout, v, theY);
		cout << endl;
	}


	combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;
	int nb_flags;

	nb_flags = sum_R;

	{
		long int *theInc;
		theInc = NEW_lint(nb_flags);

		gg->inc->geo_to_inc(v, theY, theInc, nb_flags);

		OwCF = NEW_OBJECT(combinatorics::canonical_form_classification::any_combinatorial_object);

		OwCF->init_incidence_geometry(
			theInc, nb_flags, v, gg->inc->Encoding->b, nb_flags,
			verbose_level - 2);

		FREE_lint(theInc);
	}

	if (f_v) {
		cout<< "iso_type::find_and_add_geo setting partition" << endl;
	}

	OwCF->f_partition = true;

	if (f_partition_fixing_last) {
		OwCF->partition = gg->Decomposition_with_fuse->Partition_fixing_last[v];
	}
	else {
		OwCF->partition = gg->Decomposition_with_fuse->Partition[v];
	}


	if (f_orderly) {
		if (f_v) {
			cout << "iso_type::find_and_add_geo "
					"before Canonical_forms->orderly_test" << endl;
		}
		Canonical_forms->orderly_test(
				OwCF,
				Nauty_control,
				f_new_object, verbose_level - 2);

		if (f_v) {
			cout << "iso_type::find_and_add_geo "
					"after Canonical_forms->orderly_test" << endl;
		}

		if (v == gg->inc->Encoding->v) {
			Canonical_forms->add_object(
					OwCF,
					Nauty_control,
					f_new_object, verbose_level);
		}
		else {
			FREE_OBJECT(OwCF);
		}
	}
	else {
		if (f_v) {
			cout << "iso_type::find_and_add_geo "
					"before Canonical_forms->add_object" << endl;
		}
		Canonical_forms->add_object(
				OwCF,
				Nauty_control,
				f_new_object, verbose_level);

		if (f_v) {
			cout << "iso_type::find_and_add_geo "
					"after Canonical_forms->add_object" << endl;
		}

	}

	if (f_v) {
		cout<< "iso_type::find_and_add_geo done" << endl;
	}
}


void iso_type::second()
{
	f_generate_first = true;
	f_beginning_checked = false;
}

void iso_type::set_split(
		int remainder, int modulo)
{
	f_split = true;
	split_remainder = remainder;
	split_modulo = modulo;
}


void iso_type::print_geos(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::print_geos" << endl;
	}
	{
		int h;
		long int nb_geo;

		nb_geo = Canonical_forms->Bitvector_array.size();

		cout << v << " " << gg->inc->Encoding->b << " " << sum_R << endl;
		for (h = 0; h < nb_geo; h++) {

			cout << h << " / " << nb_geo << ":" << endl;

			combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) Canonical_forms->Objects[h];


			gg->inc->print_inc(cout, v, OwCF->set);

			gg->inc->inc_to_geo(v, OwCF->set, gg->inc->Encoding->theX, sum_R);


			gg->print(cout, gg->inc->Encoding->v, v);

			cout << endl;


		}
		cout << -1 << " " << Canonical_forms->Bitvector_array.size() << endl;

		other::data_structures::tally T;
		long int *Ago;

		Ago = NEW_lint(nb_geo);
		for (h = 0; h < nb_geo; h++) {
			Ago[h] = Canonical_forms->Ago[h];
		}

		T.init_lint(Ago, nb_geo, false, 0);
		T.print_file(cout, true /* f_backwards*/);
		cout << endl;
	}
	if (f_v) {
		cout << "iso_type::print_geos done" << endl;
	}
}


void iso_type::write_inc_file(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::write_inc_file" << endl;
	}
	{
		ofstream ost(fname);
		int h;
		long int nb_geo;

		nb_geo = Canonical_forms->Bitvector_array.size();

		ost << v << " " << gg->inc->Encoding->b << " " << sum_R << endl;
		for (h = 0; h < nb_geo; h++) {

			//inc->print_geo(ost, v, theGEO1[h]);

			combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) Canonical_forms->Objects[h];
			gg->inc->print_inc(ost, v, OwCF->set);

			ost << endl;
		}
		ost << -1 << " " << Canonical_forms->Bitvector_array.size() << endl;

		other::data_structures::tally T;
		long int *Ago;

		Ago = NEW_lint(nb_geo);
		for (h = 0; h < nb_geo; h++) {
			Ago[h] = Canonical_forms->Ago[h];
		}

		T.init_lint(Ago, nb_geo, false, 0);
		T.print_file(ost, true /* f_backwards*/);
		ost << endl;
	}
	if (f_v) {
		cout << "iso_type::write_inc_file done" << endl;
	}
}

void iso_type::write_sage_file(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::write_sage_file" << endl;
	}
	{
		ofstream ost(fname);
		int h;
		long int nb_geo;

		nb_geo = Canonical_forms->Bitvector_array.size();

		ost << "# Orbiter output file " << endl;
		ost << "# Geometries with " << v << " points and " << gg->inc->Encoding->b << " blocks and " << sum_R << " flags" << endl;
		ost << "# Number of geometries = " << Canonical_forms->Bitvector_array.size() << endl;
		ost << "# Distribution of automorphism group orders: ";
		other::data_structures::tally T;
		long int *Ago;

		Ago = NEW_lint(nb_geo);
		for (h = 0; h < nb_geo; h++) {
			Ago[h] = Canonical_forms->Ago[h];
		}

		T.init_lint(Ago, nb_geo, false, 0);
		T.print_file(ost, true /* f_backwards*/);
		ost << endl;


		ost << "Geo = [";
		for (h = 0; h < nb_geo; h++) {

			//inc->print_geo(ost, v, theGEO1[h]);

			combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) Canonical_forms->Objects[h];
			gg->inc->print_sage(ost, v, OwCF->set);
			if (h < nb_geo - 1) {
				ost << ", " << endl;
			}
			//ost << endl;
		}
		ost << "]" << endl;
		//ost << -1 << " " << Canonical_forms->B.size() << endl;

		ost << endl;
		ost << "# Automorphism group orders (25 in a row):" << endl;
		ost << endl;
		ost << "Ago = [";
		for (h = 0; h < nb_geo; h++) {
			ost << Ago[h];
			if (h < nb_geo - 1) {
				ost << ", ";
			}
			if ((h + 1) % 25 == 0) {
				ost << endl;
			}
		}
		ost << "]" << endl;
	}
	if (f_v) {
		cout << "iso_type::write_sage_file done" << endl;
	}
}

void iso_type::write_blocks_file(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::write_blocks_file" << endl;
	}
	{
		ofstream ost(fname);
		int h, k;
		long int nb_geo;



		nb_geo = Canonical_forms->Bitvector_array.size();


		if (nb_geo) {
			combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

			OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) Canonical_forms->Objects[0];

			k = gg->inc->compute_k(v, OwCF->set);




			ost << v << " " << gg->inc->Encoding->b << " " << k << endl;
			for (h = 0; h < nb_geo; h++) {

				//inc->print_geo(ost, v, theGEO1[h]);


				OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) Canonical_forms->Objects[h];
				gg->inc->print_blocks(ost, v, OwCF->set);

				ost << endl;
			}
		}
		ost << -1 << " " << Canonical_forms->Bitvector_array.size() << endl;

		other::data_structures::tally T;
		long int *Ago;

		Ago = NEW_lint(nb_geo);
		for (h = 0; h < nb_geo; h++) {
			Ago[h] = Canonical_forms->Ago[h];
		}

		T.init_lint(Ago, nb_geo, false, 0);
		T.print_file(ost, true /* f_backwards*/);
		ost << endl;
	}
	if (f_v) {
		cout << "iso_type::write_blocks_file done" << endl;
	}
}

void iso_type::write_blocks_file_long(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::write_blocks_file_long" << endl;
	}
	{
		ofstream ost(fname);
		int h;
		long int nb_geo;



		nb_geo = Canonical_forms->Bitvector_array.size();


		if (nb_geo) {

			long int *theInc;
			long int *Blocks;
			int b;

			b = gg->inc->Encoding->b;

			ost << v << " " << b << endl;
			for (h = 0; h < nb_geo; h++) {

				int *K;
				int i, j, a;

				combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

				OwCF = (combinatorics::canonical_form_classification::any_combinatorial_object *) Canonical_forms->Objects[h];

				theInc = OwCF->set;

				gg->inc->compute_blocks(Blocks, K, v, theInc);

				ost << "geometry " << h << " : " << endl;

				for (i = 0; i < b; i++) {
					//ost << K[i] << " ";
					for (j = 0; j < K[i]; j++) {
						a = Blocks[i * v + j];

#if 0
						if (v == 18 && b == 39) {
							if (i >= 3) {
								if (a >= 12) {
									a -= 12;
								}
								else if (a >= 6) {
									a -= 6;
								}
							}
							a++;
						}
#endif

						ost << a;
						if (j < K[i] - 1) {
							ost << ", ";
						}
					}
					ost << "\\\\" << endl;
				}


				FREE_int(K);
				FREE_lint(Blocks);

			}
		}
		ost << -1 << " " << Canonical_forms->Bitvector_array.size() << endl;

		other::data_structures::tally T;
		long int *Ago;

		Ago = NEW_lint(nb_geo);
		for (h = 0; h < nb_geo; h++) {
			Ago[h] = Canonical_forms->Ago[h];
		}

		T.init_lint(Ago, nb_geo, false, 0);
		T.print_file(ost, true /* f_backwards*/);
		ost << endl;
	}
	if (f_v) {
		cout << "iso_type::write_blocks_file_long done" << endl;
	}
}



void iso_type::print_GEO(
		int *theY, int v, incidence *inc)
{
	//int aut_group_order;

	//aut_group_order = get_aut_group_order(pc);
	gg->print_override_theX(cout, theY, v, v);
	//cout << "automorphism group order = " << aut_group_order << endl;
}

void iso_type::print_status(
		std::ostream &ost, int f_with_flags)
{
#if 1
	ost << setw(7) << Canonical_forms->Bitvector_array.size();

#else

	if (sum_nb_GEO > 0) {
		ost << " " << setw(3) << sum_nb_GEN << " || " << setw(3) << sum_nb_TDO << " / " << setw(3) << sum_nb_GEO << " ";
	}

	ost << " " << setw(3) << nb_GEN << " || " << setw(3) << nb_TDO << " / " << setw(3) << nb_GEO << " ";
	if (f_with_flags) {
		ost << " ";
		print_flags(ost);
	}
	if (f_generate_first) {
		if (f_beginning_checked) {
			ost << " +";
		}
		else {
			ost << " -";
		}
	}
	else {
		ost << " .";
	}
	if (f_flush_line) {
		ost << " flush";
	}
#endif
}

void iso_type::print_flags(
		std::ostream &ost)
{
	ost << " ";
	if (f_split) {
		ost << " split[" << split_remainder << " % " << split_modulo << "]";
	}
}


void iso_type::print_geometry(
		inc_encoding *Encoding, int v, incidence *inc)
{
	cout << "geo" << endl;
	Encoding->print_partitioned(cout, v, v, gg, false /* f_print_isot */);
	cout << "end geo" << endl;
}




}}}}


