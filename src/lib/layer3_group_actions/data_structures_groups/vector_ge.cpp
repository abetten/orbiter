// vector_ge.cpp
//
// Anton Betten
// December 9, 2003

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


#undef PRINT_WITH_TYPE
#define RANGE_CHECKING

namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {


vector_ge::vector_ge()
{
	Record_birth();
	A = NULL;
	data = NULL;
	len = 0;
	//null();
}

vector_ge::~vector_ge()
{
	Record_death();
	if (data) {
		FREE_int(data);
		data = NULL;
	}
}

void vector_ge::null()
{
	A = NULL;
	data = NULL;
	len = 0;
}



void vector_ge::init(
		actions::action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::init" << endl;
	}
	vector_ge::A = A;
	data = NULL;
	len = 0;
	if (f_v) {
		cout << "vector_ge::init done" << endl;
	}
}

void vector_ge::copy(
		vector_ge *&vector_copy, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::copy" << endl;
	}
	if (A == NULL) {
		cout << "vector_ge::copy A == NULL" << endl;
		exit(1);
	}

	vector_copy = NEW_OBJECT(vector_ge);
	vector_copy->null();
	vector_copy->init(A, verbose_level);
	if (f_v) {
		cout << "vector_ge::copy before vector_copy->allocate" << endl;
	}
	vector_copy->allocate(len, verbose_level);
	if (f_v) {
		cout << "vector_ge::copy before loop" << endl;
	}
	for (i = 0; i < len; i++) {
		A->Group_element->element_move(ith(i), vector_copy->ith(i), 0);
	}
	if (f_v) {
		cout << "vector_ge::copy done" << endl;
	}
}

void vector_ge::init_by_hdl(
		actions::action *A,
		int *gen_hdl, int nb_gen, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::init_by_hdl" << endl;
	}
	init(A, verbose_level);
	allocate(nb_gen, verbose_level);
	for (i = 0; i < nb_gen; i++) {
		A->Group_element->element_retrieve(gen_hdl[i], ith(i), 0);
	}
	if (f_v) {
		cout << "vector_ge::init_by_hdl done" << endl;
	}
}

void vector_ge::init_by_hdl(
		actions::action *A,
		std::vector<int> &gen_hdl, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::init_by_hdl" << endl;
	}
	init(A, verbose_level);

	if (f_v) {
		cout << "vector_ge::init_by_hdl before allocate" << endl;
	}
	allocate(gen_hdl.size(), verbose_level);
	if (f_v) {
		cout << "vector_ge::init_by_hdl after allocate" << endl;
	}

	for (i = 0; i < gen_hdl.size(); i++) {
		A->Group_element->element_retrieve(gen_hdl[i], ith(i), 0);
	}
	if (f_v) {
		cout << "vector_ge::init_by_hdl done" << endl;
	}
}


void vector_ge::init_single(
		actions::action *A,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::init_single" << endl;
	}
	init(A, verbose_level);
	allocate(1, verbose_level);
	A->Group_element->element_move(Elt, ith(0), 0);
	if (f_v) {
		cout << "vector_ge::init_single done" << endl;
	}
}

void vector_ge::init_double(
		actions::action *A,
		int *Elt1, int *Elt2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::init_double" << endl;
	}
	init(A, verbose_level);
	allocate(2, verbose_level);
	A->Group_element->element_move(Elt1, ith(0), 0);
	A->Group_element->element_move(Elt2, ith(1), 0);
	if (f_v) {
		cout << "vector_ge::init_double done" << endl;
	}
}

void vector_ge::init_from_permutation_representation(
		actions::action *A, groups::sims *S, int *data,
	int nb_elements, int verbose_level)
// data[nb_elements * A->degree]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int *Elt;

	if (f_v) {
		cout << "vector_ge::init_from_permutation_representation" << endl;
		cout << "vector_ge::init_from_permutation_representation A = ";
		A->print_info();
		cout << endl;
	}
	Elt = NEW_int(A->elt_size_in_int);
	init(A, verbose_level);
	allocate(nb_elements, verbose_level);
	for (i = 0; i < nb_elements; i++) {
		if (f_v) {
			cout << "vector_ge::init_from_permutation_representation "
					"i = " << i << " / " << nb_elements << endl;
		}
		A->Group_element->make_element_from_permutation_representation(
				Elt, S, data + i * A->degree, verbose_level - 1);
		if (f_vv) {
			cout << "vector_ge::init_from_permutation_representation "
					"generator " << i << ": " << endl;
			A->Group_element->element_print_quick(Elt, cout);
			A->Group_element->element_print_latex(Elt, cout);
		}
		A->Group_element->element_move(Elt, ith(i), 0);
	}
	
	FREE_int(Elt);
	if (f_v) {
		cout << "vector_ge::init_from_permutation_representation done" << endl;
	}
}

void vector_ge::init_from_data(
		actions::action *A, int *data,
	int nb_elements, int elt_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;
	int *Elt;

	if (f_v) {
		cout << "vector_ge::init_from_data" << endl;
	}

	if (elt_size != A->make_element_size) {
		cout << "vector_ge::init_from_data "
				"elt_size != A->make_element_size" << endl;
		cout << "elt_size = " << elt_size << endl;
		cout << "A->make_element_size = " << A->make_element_size << endl;
		exit(1);
	}
	Elt = NEW_int(A->elt_size_in_int);
	init(A, verbose_level - 2);
	allocate(nb_elements, verbose_level - 2);
	for (i = 0; i < nb_elements; i++) {
		if (f_v) {
			cout << "vector_ge::init_from_data i = " << i << " / " << nb_elements << endl;
		}

		if (f_v) {
			cout << "vector_ge::init_from_data "
					"data      " << i << ": ";
			Int_vec_print(cout, data + i * elt_size, elt_size);
			cout << endl;
			A->Group_element->element_print_quick(Elt, cout);
		}

		A->Group_element->make_element(
				Elt, data + i * elt_size, 0 /*verbose_level - 2*/);
		if (f_v) {
			cout << "vector_ge::init_from_data "
					"generator " << i << ": " << endl;
			A->Group_element->element_print_quick(Elt, cout);
		}
		A->Group_element->element_move(Elt, ith(i), 0);
	}
	
	FREE_int(Elt);
	if (f_v) {
		cout << "vector_ge::init_from_data done" << endl;
	}
}

void vector_ge::init_transposed(
		vector_ge *v,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "vector_ge::init_transposed" << endl;
	}

	init(v->A, verbose_level - 2);
	allocate(v->len, verbose_level - 2);
	for (i = 0; i < v->len; i++) {
		if (f_v) {
			cout << "before element_transpose " << i << " / "
					<< v->len << ":" << endl;
			A->Group_element->element_print_quick(v->ith(i), cout);
		}
		A->Group_element->element_transpose(v->ith(i), ith(i),
				0 /* verbose_level*/);
		if (f_v) {
			cout << "after element_transpose " << i << " / "
					<< v->len << ":" << endl;
			A->Group_element->element_print_quick(ith(i), cout);
		}
	}

	if (f_v) {
		cout << "vector_ge::init_transposed done" << endl;
	}
}


void vector_ge::init_conjugate_svas_of(
		vector_ge *v,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *Elt1, *Elt2, *Elt3;

	if (f_v) {
		cout << "vector_ge::init_conjugate_svas_of" << endl;
	}

	init(v->A, verbose_level);
	allocate(v->len, verbose_level);

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	A->Group_element->invert(Elt, Elt1);
	for (i = 0; i < len; i++) {
		A->Group_element->element_mult(Elt1, v->ith(i), Elt2, false);
		A->Group_element->element_mult(Elt2, Elt, Elt3, false);
		A->Group_element->element_move(Elt3, ith(i), false);
	}
	
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_v) {
		cout << "vector_ge::init_conjugate_svas_of done" << endl;
	}
}

void vector_ge::init_conjugate_sasv_of(
		vector_ge *v,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *Elt1, *Elt2, *Elt3;

	if (f_v) {
		cout << "vector_ge::init_conjugate_svas_of" << endl;
	}

	init(v->A, verbose_level);
	allocate(v->len, verbose_level);

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	A->Group_element->invert(Elt, Elt1);
	for (i = 0; i < len; i++) {
		A->Group_element->element_mult(Elt, v->ith(i), Elt2, false);
		A->Group_element->element_mult(Elt2, Elt1, Elt3, false);
		A->Group_element->element_move(Elt3, ith(i), false);
	}
	
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_v) {
		cout << "vector_ge::init_conjugate_sasv_of done" << endl;
	}
}

vector_ge *vector_ge::make_inverses(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	vector_ge *v;

	if (f_v) {
		cout << "vector_ge::make_inverses" << endl;
	}

	v = NEW_OBJECT(vector_ge);
	v->init(A, verbose_level);
	v->allocate(len, verbose_level);

	for (i = 0; i < len; i++) {
		A->Group_element->invert(ith(i), v->ith(i));
	}
	if (f_v) {
		cout << "vector_ge::make_inverses done" << endl;
	}
	return v;
}


int *vector_ge::ith(
		int i)
{
#ifdef RANGE_CHECKING
	if (i < 0 || i >= len) {
		cout << "vector_ge::ith() access error "
				"i = " << i << " len = " << len << endl;
		exit(1);
		}
#endif
	return data + i * A->elt_size_in_int; 
}

void vector_ge::print(
		std::ostream &ost)
{
	int i;

	for (i = 0; i < len; i++) {
		ost << "Element " << i << " / " << len << " is:" << endl;
		A->Group_element->element_print_quick(ith(i), ost);
		ost << endl;
	}
}

void vector_ge::print_quick(
		std::ostream& ost)
{
	int i;
	
	ost << "(" << endl;
	//ost << "len=" << len << " A->elt_size_in_int=" << A->elt_size_in_int << " data=" << data << endl;
	for (i = 0; i < len; i++) {
		if (data == NULL) {
			cout << "vector_ge::print fatal: data == NULL" << endl;
			exit(1);
		}
		A->Group_element->element_print_quick(ith(i), ost);
		if (i < len - 1) {
			ost << ", " << endl;
		}
	}
	ost << ")" << endl;
}

void vector_ge::print_tex(
		std::ostream &ost)
{
	int i;

	ost << "$$" << endl;
	for (i = 0; i < len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A->Group_element->element_print_latex(ith(i), ost);
		if (i < len - 1) {
			ost << ", " << endl;
		}
		if (((i + 1) % 3) == 0 && i < len - 1) {
			ost << "$$" << endl;
			ost << "$$" << endl;
			}
		}
	ost << "$$" << endl;
}


void vector_ge::report_elements(
		std::string &label,
		int f_with_permutation,
		int f_override_action, actions::action *A_special,
		std::string &options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::report_elements" << endl;
	}


	std::map<std::string, std::string> symbol_table;

	other::data_structures::string_tools ST;

	int f_dense = false;

	if (options.length()) {

		ST.parse_value_pairs(symbol_table,
				options, verbose_level - 1);


		{
			std::map<std::string, std::string>::iterator it = symbol_table.begin();


			// Iterate through the map and print the elements
			while (it != symbol_table.end()) {
				string label;
				string val;

				label = it->first;
				val = it->second;
				//std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
				//assignment.insert(std::make_pair(label, a));
				if (ST.stringcmp(label, "dense") == 0) {
					f_dense = true;;
				}
				++it;
			}
		}
	}



	other::orbiter_kernel_system::file_io Fio;


	int *Elt;
	algebra::ring_theory::longinteger_object go;


	string fname;

	fname = label + "_elements.tex";


	{
		ofstream ost(fname);
		other::l1_interfaces::latex_interface L;
		L.head_easy(ost);


		int *Order;
		int ord;
		actions::action *A1;

		Order = NEW_int(len);


		if (f_override_action) {
			A1 = A_special;
		}
		else {
			A1 = A;
		}


		if (f_dense) {

			int i;

			for (i = 0; i < len; i++) {

				Elt = ith(i);

				//ord = A1->Group_element->element_order(Elt);
				//ost << "Element " << setw(5) << i << " / "
				//		<< len << " of order " << ord << ":" << endl;

				ost << "$" << endl;
				A1->Group_element->element_print_latex(Elt, ost);
				ost << "$" << endl;

#if 0
				A1->print_one_element_tex(
						ost,
						Elt, f_with_permutation);
#endif

				if (i < len - 1) {
					ost << ", ";
				}
				ost << endl;
				//Order[i] = ord;
			}

		}
		else {
			ost << "Group elements in action ";
			ost << "$";
			ost << A1->label_tex;
			ost << "$\\\\" << endl;

			int i;

			for (i = 0; i < len; i++) {

				if (f_v) {
					cout << "Element " << i << " / " << len << ":" << endl;
				}

				Elt = ith(i);

				ord = A1->Group_element->element_order(Elt);
				ost << "Element " << setw(5) << i << " / "
						<< len << " of order " << ord << ":" << endl;


				ost << "$$" << endl;
				A1->Group_element->element_print_latex(Elt, ost);
				ost << "$$" << endl;

				Int_vec_print_bare_fully(ost, Elt, A->make_element_size);
				ost << "\\\\" << endl;


#if 0
				A1->print_one_element_tex(
						ost,
						Elt, f_with_permutation);
#endif

				Order[i] = ord;
			}
			other::data_structures::tally T;

			T.init(Order, len, false, 0 /*verbose_level*/);

			ost << "Order structure:\\\\" << endl;
			ost << "$" << endl;
			T.print_file_tex_we_are_in_math_mode(ost, true /* f_backwards */);
			ost << "$" << endl;
			ost << "\\\\" << endl;



			FREE_int(Order);
		}

		L.foot(ost);

	}
	if (f_v) {
		cout << "vector_ge::report_elements "
			"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "vector_ge::report_elements done" << endl;
	}
}


void vector_ge::report_elements_coded(
		std::string &label,
		std::string &fname_out,
		int f_override_action, actions::action *A_special,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::report_elements_coded" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;


	int *Elt;
	algebra::ring_theory::longinteger_object go;


	//string fname_out;

	fname_out = label + "_elements.tex";


	{
		ofstream ost(fname_out);
		other::l1_interfaces::latex_interface L;
		L.head_easy(ost);


		actions::action *A1;


		if (f_override_action) {
			A1 = A_special;
		}
		else {
			A1 = A;
		}

		int i;

		for (i = 0; i < len; i++) {

			Elt = ith(i);

			A1->Group_element->print_for_make_element(
					ost, Elt);
			ost << "\\\\" << endl;
		}



		L.foot(ost);

	}
	if (f_v) {
		cout << "vector_ge::report_elements_coded "
			"Written file " << fname_out << " of size "
			<< Fio.file_size(fname_out) << endl;
	}


	if (f_v) {
		cout << "vector_ge::report_elements_coded done" << endl;
	}
}




void vector_ge::print_generators_tex(
		algebra::ring_theory::longinteger_object &go, std::ostream &ost)
{
	int i;

	ost << "Generators for a group of order " << go << ":" << endl;
	ost << "$$" << endl;
	for (i = 0; i < len; i++) {
		//cout << "Generator " << i << " / " << gens->len
		// << " is:" << endl;
		A->Group_element->element_print_latex(ith(i), ost);
		if (((i + 1) % 4) == 0 && i < len - 1) {
			ost << "$$" << endl;
			ost << "$$" << endl;
		}
	}
	ost << "$$" << endl;
	for (i = 0; i < len; i++) {
		A->Group_element->element_print_for_make_element(ith(i), ost);
		ost << "\\\\" << endl;
	}
}

void vector_ge::print_as_permutation(
		std::ostream& ost)
{
	int i;
	
	ost << "(" << endl;
	for (i = 0; i < len; i++) {
		A->Group_element->element_print(ith(i), ost);
		if (A->degree < 1000) {
			cout << "is the permutation" << endl;
			A->Group_element->element_print_as_permutation(ith(i), ost);
		}
		else {
			cout << "vector_ge::print_as_permutation "
					"the degree is too large, we won't print "
					"the permutation" << endl;
		}
		if (i < len - 1) {
			ost << ", " << endl;
		}
	}
	ost << ")" << endl;
}

void vector_ge::allocate(
		int length, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::allocate" << endl;
		cout << "vector_ge::allocate length = " << length << endl;
	}
	if (f_v) {
		cout << "vector_ge::allocate A = " << endl;
		A->print_info();
	}
	if (data) {
		if (f_v) {
			cout << "vector_ge::allocate before FREE-int(data)" << endl;
		}
		FREE_int(data);
		//cout << "vector_ge::allocate warning, data != NULL, "
		//"we seem to be having a memory leak here" << endl;
	}
	len = length;
	data = NEW_int(length * A->elt_size_in_int);
	if (f_v) {
		cout << "vector_ge::allocate done" << endl;
	}
}

void vector_ge::reallocate(
		int new_length, int verbose_level)
{
	int *data2;
	int *elt, *elt2, i, l;
	
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::reallocate" << endl;
	}
	data2 = NEW_int(new_length * A->elt_size_in_int);

	l = MINIMUM(len, new_length);
	for (i = 0; i < l; i++) {
		elt = ith(i);
		elt2 = data2 + i * A->elt_size_in_int;
		A->Group_element->element_move(elt, elt2, false);
	}
	if (data) {
		FREE_int(data);
		data = NULL;
	}
	data = data2;
	len = new_length;
	if (f_v) {
		cout << "vector_ge::reallocate done" << endl;
	}
}

void vector_ge::reallocate_and_insert_at(
		int position, int *elt, int verbose_level)
{
	int *data2;
	int *elt1, *elt2, i;
	
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::reallocate_and_insert_at len=" << len << endl;
	}
	data2 = NEW_int((len + 1) * A->elt_size_in_int);
	for (i = 0; i < len; i++) {
		elt1 = ith(i);
		if (i >= position) {
			elt2 = data2 + (i + 1) * A->elt_size_in_int;
		}
		else {
			elt2 = data2 + i * A->elt_size_in_int;
		}
		A->Group_element->element_move(elt1, elt2, false);
	}
	if (data) {
		FREE_int(data);
		data = NULL;
	}
	else {
		if (f_v) {
			cout << "vector_ge::reallocate_and_insert_at data == NULL" << endl;
		}
	}
	data = data2;
	len = len + 1;
	if (position < 0 || position >= len) {
		cout << "vector_ge::reallocate_and_insert_at "
				"position out of bounds, position=" << position << endl;
		exit(1);
	}
	copy_in(position, elt);
	if (f_v) {
		cout << "vector_ge::reallocate_and_insert_at done" << endl;
	}
}

void vector_ge::insert_at(
		int length_before,
		int position, int *elt, int verbose_level)
// does not reallocate, but shifts elements up to make space.
// the last element might be lost if there is no space.
{
	int *elt1, *elt2, i;
	
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::insert_at" << endl;
	}
	for (i = length_before; i >= position; i--) {
		if (i + 1 >= len) {
			continue;
		}
		
		elt1 = ith(i);
		elt2 = ith(i + 1);
		A->Group_element->element_move(elt1, elt2, false);
	}
	copy_in(position, elt);
	if (f_v) {
		cout << "vector_ge::insert_at done" << endl;
	}
}

void vector_ge::append(
		int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::append" << endl;
	}
	reallocate_and_insert_at(len, elt, verbose_level);
	if (f_v) {
		cout << "vector_ge::append done" << endl;
	}
}

void vector_ge::copy_in(
		int i, int *elt)
{
	int *elt2 = ith(i);
	A->Group_element->element_move(elt, elt2, false);
};

void vector_ge::copy_out(
		int i, int *elt)
{
	int *elt2 = ith(i);
	A->Group_element->element_move(elt2, elt, false);
}

void vector_ge::conjugate_svas(
		int *Elt)
{
	int i;
	int *Elt1, *Elt2, *Elt3;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	A->Group_element->invert(Elt, Elt1);
	for (i = 0; i < len; i++) {
		A->Group_element->element_mult(Elt1, ith(i), Elt2, false);
		A->Group_element->element_mult(Elt2, Elt, Elt3, false);
		A->Group_element->element_move(Elt3, ith(i), false);
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
}

void vector_ge::conjugate_sasv(
		int *Elt)
{
	int i;
	int *Elt1, *Elt2, *Elt3;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	A->Group_element->invert(Elt, Elt1);
	for (i = 0; i < len; i++) {
		A->Group_element->element_mult(Elt, ith(i), Elt2, false);
		A->Group_element->element_mult(Elt2, Elt1, Elt3, false);
		A->Group_element->element_move(Elt3, ith(i), false);
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
}

void vector_ge::print_with_given_action(
		std::ostream &ost, actions::action *A2)
{
	int i, l;

	l = len;
	for (i = 0; i < l; i++) {
		ost << "generator " << i << ":" << endl;
		A->Group_element->element_print_quick(ith(i), ost);
		ost << endl;
		A2->Group_element->element_print_as_permutation(ith(i), ost);
		ost << endl;
	}
}

void vector_ge::print(
		std::ostream &ost,
		int f_print_as_permutation,
	int f_offset, int offset,
	int f_do_it_anyway_even_for_big_degree,
	int f_print_cycles_of_length_one,
	int verbose_level)
{
	int i, l;
	
	l = len;
	if (!f_offset) {
		offset = 0;
	}
	ost << "vector of " << l << " group elements:" << endl;
	ost << "f_print_as_permutation=" << f_print_as_permutation << endl;
	for (i = 0; i < l; i++) {
		ost << "generator " << i << " / " << l << ":" << endl;
		A->Group_element->element_print_quick(ith(i), ost);
		ost << endl;
		if (f_print_as_permutation) {
			//A->element_print_as_permutation(ith(i), ost);
			A->Group_element->element_print_as_permutation_with_offset(ith(i), ost,
				offset, f_do_it_anyway_even_for_big_degree, 
				f_print_cycles_of_length_one, verbose_level - 1);
			ost << endl;
		}
	}
}

void vector_ge::print_for_make_element(
		std::ostream &ost)
{
	int i, l;

	l = len;
	ost << "vector of length " << l << ":" << endl;
	for (i = 0; i < l; i++) {
		A->Group_element->element_print_for_make_element(ith(i), ost);
		ost << "\\\\" << endl;
	}
}


void vector_ge::write_to_memory_object(
		other::orbiter_kernel_system::memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "vector_ge::write_to_memory_object" << endl;
	}
	m->write_int(len);
	for (i = 0; i < len; i++) {
		A->Group_element->element_write_to_memory_object(ith(i), m, 0);
	}
}

void vector_ge::read_from_memory_object(
		other::orbiter_kernel_system::memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;

	if (f_v) {
		cout << "vector_ge::read_from_memory_object" << endl;
	}
	m->read_int(&l);
	allocate(l, verbose_level);
	for (i = 0; i < len; i++) {
		A->Group_element->element_read_from_memory_object(ith(i), m, 0);
	}
}

void vector_ge::write_to_file_binary(
		std::ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "vector_ge::write_to_file_binary" << endl;
	}
	fp.write((char *) &len, sizeof(int));
	for (i = 0; i < len; i++) {
		if (f_v) {
			cout << "vector_ge::write_to_file_binary writing element " << i << " / " << len << endl;
		}
		A->Group_element->element_write_to_file_binary(ith(i), fp, verbose_level);
	}
}

void vector_ge::read_from_file_binary(
		std::ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;

	if (f_v) {
		cout << "vector_ge::read_from_file_binary" << endl;
	}
	fp.read((char *) &l, sizeof(int));
	allocate(l, verbose_level);
	for (i = 0; i < len; i++) {
		if (f_v) {
			cout << "vector_ge::read_from_file_binary reading element " << i << " / " << len << endl;
		}
		A->Group_element->element_read_from_file_binary(ith(i), fp, verbose_level);
	}
}

void vector_ge::write_to_csv_file_coded(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "vector_ge::write_to_csv_file_coded" << endl;
	}
	int *Table;

	Table = NEW_int(len * A->make_element_size);
	for (i = 0; i < len; i++) {
		for (j = 0; j < A->make_element_size; j++) {
			Table[i * A->make_element_size + j] = ith(i)[j];
		}
	}
	other::orbiter_kernel_system::file_io Fio;

	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Table, len, A->make_element_size);
	if (f_v) {
		cout << "vector_ge::write_to_csv_file_coded written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "vector_ge::write_to_csv_file_coded done" << endl;
	}
}

void vector_ge::save_csv(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::save_csv" << endl;
	}

	int i;
	int *Elt;
	int *data;
	int nb_rows, nb_cols;
	std::string *Table;
	std::string *Col_headings;

	nb_cols = 2;
	nb_rows = len;
	Table = new std::string [nb_rows * nb_cols];
	Col_headings = new std::string [nb_cols];

	Col_headings[0] = "Row";
	Col_headings[1] = "Element";

	data = NEW_int(A->make_element_size);

	for (i = 0; i < len; i++) {
		Elt = ith(i);

		A->Group_element->element_code_for_make_element(Elt, data);

		Table[nb_cols * i + 0] = std::to_string(i);
		Table[nb_cols * i + 1] = "\"" + Int_vec_stringify(data, A->make_element_size) + "\"";
	}

	other::orbiter_kernel_system::file_io Fio;

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);


	delete [] Col_headings;
	delete [] Table;
	FREE_int(data);

	if (f_v) {
		cout << "vector_ge::save_csv Written file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

}

void vector_ge::export_inversion_graphs(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::export_inversion_graphs" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	int h;
	int *Elt;
	int *perm;
	int N2;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	N2 = Combi.int_n_choose_k(A->degree, 2);

	perm = NEW_int(A->degree);

	{
		ofstream ost(fname);

		ost << "Row";
		for (h = 0; h < N2; h++) {
			ost << ",C" << h;
		}
		ost << endl;
		for (h = 0; h < len; h++) {
			Elt = ith(h);


			int *Adj;
			int N;
			int i, j;

			A->Group_element->compute_permutation(
					Elt, perm, 0 /* verbose_level*/);

			combinatorics::graph_theory::graph_theory_domain GT;


			if (f_v) {
				cout << "vector_ge::export_inversion_graphs "
						"before GT.make_inversion_graph" << endl;
			}
			GT.make_inversion_graph(Adj, N, perm, A->degree, verbose_level);
			if (f_v) {
				cout << "vector_ge::export_inversion_graphs "
						"after GT.make_inversion_graph" << endl;
			}

			ost << h;
			for (i = 0; i < N; i++) {
				for (j = i + 1; j < N; j++) {
					ost << "," << Adj[i * N + j];
				}
			}
			ost << endl;


			FREE_int(Adj);


		}
		ost << "END" << endl;
	}
	FREE_int(perm);

	if (f_v) {
		cout << "sims::export_inversion_graphs "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

}


void vector_ge::read_column_csv(
		std::string &fname,
		actions::action *A, int col_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::read_column_csv" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	other::data_structures::spreadsheet S;
	int n, i, me_sz;


	init(A, verbose_level);

	me_sz = A->make_element_size;

	if (f_v) {
		cout << "vector_ge::read_column_csv reading file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

	S.read_spreadsheet(fname, verbose_level);

	n = S.nb_rows - 1;

	allocate(n, verbose_level);
	for (i = 0; i < n; i++) {
		string s;
		int *data;
		int sz;

		S.get_string(s, i + 1, col_idx);
		Int_vec_scan(s, data, sz);
		if (sz != me_sz) {
			cout << "vector_ge::read_column_csv sz != me_sz" << endl;
			cout << "vector_ge::read_column_csv sz = " << sz << endl;
			cout << "vector_ge::read_column_csv me_sz = " << me_sz << endl;
			exit(1);
		}
		A->Group_element->make_element(ith(i), data, 0 /*verbose_level*/);
		FREE_int(data);
	}

	if (f_v) {
		cout << "vector_ge::read_column_csv done" << endl;
	}
}

void vector_ge::read_column_csv_using_column_label(
		std::string &fname,
		actions::action *A,
		std::string &column_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::read_column_csv column_label = " << column_label << endl;
	}


	other::orbiter_kernel_system::file_io Fio;
	other::data_structures::spreadsheet S;
	int n, i, me_sz;


	init(A, verbose_level);

	me_sz = A->make_element_size;

	if (f_v) {
		cout << "vector_ge::read_column_csv reading file " << fname
				<< " of size " << Fio.file_size(fname) << endl;
	}

	S.read_spreadsheet(fname, verbose_level);


	int col_idx;

	col_idx = S.find_column(column_label);


	n = S.nb_rows - 1;

	allocate(n, verbose_level);
	for (i = 0; i < n; i++) {
		string s;
		int *data;
		int sz;

		S.get_string(s, i + 1, col_idx);

		if (s.length() >= 2 && s[0] == '"' && s[s.length() - 1] == '"') {
			string s2;

			s2 = s.substr(1, s.length() - 2);
			s = s2;
		}
		Int_vec_scan(s, data, sz);
		if (sz != me_sz) {
			cout << "vector_ge::read_column_csv sz != me_sz" << endl;
			cout << "vector_ge::read_column_csv sz = " << sz << endl;
			cout << "vector_ge::read_column_csv me_sz = " << me_sz << endl;
			exit(1);
		}
		A->Group_element->make_element(ith(i), data, 0 /*verbose_level*/);
		FREE_int(data);
	}

	if (f_v) {
		cout << "vector_ge::read_column_csv done" << endl;
	}
}



void vector_ge::extract_subset_of_elements_by_rank_text_vector(
		std::string &rank_vector_text, groups::sims *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::extract_subset_of_elements_"
				"by_rank_text_vector" << endl;
	}
	int *v;
	int len;

	Int_vec_scan(rank_vector_text, v, len);
	if (f_v) {
		cout << "vector_ge::extract_subset_of_elements_"
				"by_rank_text_vector after scanning: ";
		Int_vec_print(cout, v, len);
		cout << endl;
	}
	extract_subset_of_elements_by_rank(v, len, S, verbose_level);
	FREE_int(v);
	if (f_v) {
		cout << "vector_ge::extract_subset_of_elements_"
				"by_rank_text_vector done" << endl;
	}
}

void vector_ge::extract_subset_of_elements_by_rank(
		int *rank_vector, int len, groups::sims *S,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, r;
	int *Elt;

	if (f_v) {
		cout << "vector_ge::extract_subset_of_elements_by_rank" << endl;
	}
	Elt = NEW_int(A->elt_size_in_int);
	allocate(len, verbose_level);
	for (i = 0; i < len; i++) {
		r = rank_vector[i];
		S->element_unrank_lint(r, Elt);

		if (f_v) {
			cout << "vector_ge::extract_subset_of_elements_by_rank "
					"element " << i << " = " << r << " / " << len << endl;
			A->Group_element->element_print_quick(Elt, cout);
		}
		A->Group_element->element_move(Elt, ith(i), 0);
	}
	FREE_int(Elt);
	if (f_v) {
		cout << "vector_ge::extract_subset_of_elements_by_rank done" << endl;
	}
}

void vector_ge::compute_rank_vector(
		long int *&rank_vector, groups::sims *Sims,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i;
	int *Elt;

	if (f_v) {
		cout << "vector_ge::compute_rank_vector" << endl;
	}

	rank_vector = NEW_lint(len);
	Elt = NEW_int(A->elt_size_in_int);
	for (i = 0; i < len; i++) {

		A->Group_element->element_move(ith(i), Elt, 0);

		rank_vector[i] = Sims->element_rank_lint(Elt);

		if (f_v) {
			cout << "vector_ge::compute_rank_vector "
					"element " << i << " / " << len << " has rank " << rank_vector[i] << endl;
			A->Group_element->element_print_quick(Elt, cout);
		}
	}
	FREE_int(Elt);
	if (f_v) {
		cout << "vector_ge::compute_rank_vector done" << endl;
	}
}



int vector_ge::test_if_all_elements_stabilize_a_point(
		actions::action *A2, int pt)
{
	int i;
	
	for (i = 0; i < len; i++) {
		if (A2->Group_element->element_image_of(pt, ith(i), 0) != pt) {
			return false;
		}
	}
	return true;
}

int vector_ge::test_if_all_elements_stabilize_a_set(
		actions::action *A2,
		long int *set, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "vector_ge::test_if_all_elements_stabilize_a_set" << endl;
	}
	
	for (i = 0; i < len; i++) {
		if (f_v) {
			cout << "testing element " << i << " / " << len << endl;
		}
		if (!A2->Group_element->test_if_set_stabilizes(ith(i),
				sz, set, 0 /* verbose_level*/)) {
			return false;
		}
	}
	if (f_v) {
		cout << "vector_ge::test_if_all_elements_stabilize_a_set done" << endl;
	}
	return true;
}


groups::schreier *vector_ge::compute_all_point_orbits_schreier(
		actions::action *A_given,
		int print_interval,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::schreier *Sch;

	if (f_v) {
		cout << "vector_ge::compute_all_point_orbits_schreier "
				"degree = " << A_given->degree << endl;
	}
	if (f_v) {
		cout << "vector_ge::compute_all_point_orbits_schreier "
				"action ";
		A_given->print_info();
		cout << endl;
	}


	Sch = NEW_OBJECT(groups::schreier);

	Sch->init(A_given, verbose_level - 2);
	//Sch->initialize_tables();
	Sch->Generators_and_images->init_generators(
			*this, verbose_level - 2);
	if (f_v) {
		cout << "vector_ge::compute_all_point_orbits_schreier "
				"before Sch->compute_all_point_orbits" << endl;
	}
	Sch->compute_all_point_orbits(print_interval, verbose_level);
	if (f_v) {
		cout << "vector_ge::compute_all_point_orbits_schreier "
				"after Sch->compute_all_point_orbits" << endl;
	}

	if (f_v) {
		cout << "vector_ge::compute_all_point_orbits_schreier "
				"we found " << Sch->Forest->nb_orbits << " orbits" << endl;
	}

	if (f_v) {
		cout << "vector_ge::compute_all_point_orbits_schreier done" << endl;
	}
	return Sch;
}

void vector_ge::reverse_isomorphism_exterior_square(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	algebra::field_theory::finite_field *F;

	if (f_v) {
		cout << "vector_ge::reverse_isomorphism_exterior_square" << endl;
	}

	geometry::projective_geometry::klein_correspondence *K;
	geometry::orthogonal_geometry::orthogonal *O;
	int A4[17];


	F = A->matrix_group_finite_field();

	O = NEW_OBJECT(geometry::orthogonal_geometry::orthogonal);
	O->init(1 /* epsilon */, 6 /* n */, F, verbose_level);

	K = NEW_OBJECT(geometry::projective_geometry::klein_correspondence);
	K->init(F, O, verbose_level);


	for (i = 0; i < len; i++) {

		if (f_v) {
			cout << "vector_ge::reverse_isomorphism_exterior_square "
					"generator " << i << " / " << len << ":" << endl;
		}


		//K->reverse_isomorphism(ith(i), A4, verbose_level);

		int f_has_polarity;

		K->reverse_isomorphism_with_polarity(ith(i), A4, f_has_polarity, verbose_level - 2);

		if (f_v) {
			cout << "vector_ge::reverse_isomorphism_exterior_square "
					"before:" << endl;
			Int_matrix_print(ith(i), 6, 6);

			cout << "vector_ge::reverse_isomorphism_exterior_square "
					"after:" << endl;
			Int_matrix_print(A4, 4, 4);
			cout << "vector_ge::reverse_isomorphism_exterior_square "
					"f_has_polarity = " << f_has_polarity << endl;
		}

	}

	FREE_OBJECT(K);
	FREE_OBJECT(O);
	if (f_v) {
		cout << "vector_ge::reverse_isomorphism_exterior_square done" << endl;
	}
}

void vector_ge::matrix_representation(
		induced_actions::action_on_homogeneous_polynomials *A_on_HPD,
		int *&M, int &nb_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int n;

	if (f_v) {
		cout << "vector_ge::matrix_representation" << endl;
	}

	nb_gens = len;

	n = A_on_HPD->dimension;

	M = NEW_int(len * n * n);

	for (i = 0; i < len; i++) {

		A_on_HPD->compute_representation(ith(i),
				M + i * n * n, verbose_level);

	}

	if (f_v) {
		cout << "vector_ge::matrix_representation done" << endl;
	}
}

void vector_ge::stab_BLT_set_from_catalogue(
		actions::action *A,
		algebra::field_theory::finite_field *F, int iso,
	std::string &target_go_text,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::stab_BLT_set_from_catalogue" << endl;
		cout << "q=" << F->q << endl;
		cout << "iso=" << iso << endl;
		cout << "A=" << endl;
		A->print_info();
	}

	int *data;
	int nb_gens;
	int data_size;
	//string ascii_target_go;
	//ring_theory::longinteger_object target_go;
	int i;
	combinatorics::knowledge_base::knowledge_base K;

	if (f_v) {
		cout << "vector_ge::stab_BLT_set_from_catalogue "
				"before K.BLT_stab_gens" << endl;
	}
	K.BLT_stab_gens(F->q, iso, data, nb_gens, data_size, target_go_text);

	if (f_v) {
		cout << "vector_ge::stab_BLT_set_from_catalogue "
				"data_size=" << data_size << endl;
		cout << "vector_ge::stab_BLT_set_from_catalogue "
				"nb_gens=" << nb_gens << endl;
		cout << "vector_ge::stab_BLT_set_from_catalogue "
				"target_go_text=" << target_go_text << endl;
	}

	init(A, verbose_level - 2);
	//target_go.create_from_base_10_string(target_go_text);


	allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		A->Group_element->make_element(ith(i), data + i * data_size, 0);
	}

	if (f_v) {
		cout << "vector_ge::stab_BLT_set_from_catalogue "
				"generators are:" << endl;
		print_quick(cout);
	}

	if (f_v) {
		cout << "vector_ge::stab_BLT_set_from_catalogue done" << endl;
	}
}

int vector_ge::test_if_in_set_stabilizer(
		actions::action *A,
		long int *set, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;
	int i, c;

	if (f_v) {
		cout << "vector_ge::test_if_in_set_stabilizer" << endl;
	}

	ret = true;
	for (i = 0; i < len; i++) {
		c = A->Group_element->check_if_in_set_stabilizer(
				ith(i),
				sz, set, 0 /* verbose_level*/);
		if (!c) {
			if (f_v) {
				cout << "vector_ge::test_if_in_set_stabilizer element "
						<< i << " fails to stabilize the given set" << endl;
			}
			ret = false;
			break;
		}
	}
	if (f_v) {
		cout << "vector_ge::test_if_in_set_stabilizer done" << endl;
	}
	return ret;
}


void vector_ge::print_generators_gap(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "vector_ge::print_generators_gap" << endl;
	}
	ost << "#Generators in GAP format are:" << endl;
	if (A->degree < 1000) {
		ost << "G := Group([";
		for (i = 0; i < len; i++) {
			if (f_v) {
				cout << "vector_ge::print_generators_gap "
						"i=" << i << " / " << len << endl;
			}
			A->Group_element->element_print_as_permutation_with_offset(
					ith(i), ost,
					1 /*offset*/,
					true /* f_do_it_anyway_even_for_big_degree */,
					false /* f_print_cycles_of_length_one */,
					0 /* verbose_level*/);
			if (i < len - 1) {
				ost << ", " << endl;
			}
		}
		ost << "]);" << endl;
	}
	else {
		ost << "too big to print" << endl;
	}
	if (f_v) {
		cout << "vector_ge::print_generators_gap done" << endl;
	}
}


void vector_ge::print_generators_gap_in_different_action(
		std::ostream &ost, actions::action *A2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::print_generators_gap_in_different_action" << endl;
	}
	int i;

	ost << "Generators in GAP format are:" << endl;
	if (A->degree < 200) {
		ost << "G := Group([";
		for (i = 0; i < len; i++) {
			A2->Group_element->element_print_as_permutation_with_offset(
					ith(i), ost,
					1 /*offset*/,
					true /* f_do_it_anyway_even_for_big_degree */,
					false /* f_print_cycles_of_length_one */,
					0 /* verbose_level*/);
			if (i < len - 1) {
				ost << ", " << endl;
			}
		}
		ost << "]);" << endl;
	}
	else {
		ost << "too big to print" << endl;
	}
	if (f_v) {
		cout << "vector_ge::print_generators_gap_in_different_action done" << endl;
	}
}


void vector_ge::print_generators_compact(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::print_generators_compact" << endl;
	}
	int i, j, a;

	ost << "Generators in compact permutation form are:" << endl;
	if (A->degree < 200) {
		ost << len << " " << A->degree << endl;
		for (i = 0; i < len; i++) {
			for (j = 0; j < A->degree; j++) {
				a = A->Group_element->element_image_of(
						j,
						ith(i),
						0 /* verbose_level */);
				ost << a << " ";
				}
			ost << endl;
			}
		ost << "-1" << endl;
	}
	else {
		ost << "too big to print" << endl;
	}
	if (f_v) {
		cout << "vector_ge::print_generators_compact done" << endl;
	}
}

void vector_ge::multiply_with(
		vector_ge **V, int nb_with, vector_ge *&result, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::multiply_with" << endl;
	}


	int sz, h;

	sz = len;

	for (h = 0; h < nb_with; h++) {
		if (V[h]->len != sz) {
			cout << "vector_ge::multiply_with the vectors must all have the same length" << endl;
			exit(1);
		}
	}


	result = NEW_OBJECT(vector_ge);
	result->init(
			A, 0 /* verbose_level*/);
	result->allocate(
			len, 0 /* verbose_level*/);
	int i;
	int *Elt1;
	int *Elt2;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	for (i = 0; i < len; i++) {

		A->Group_element->element_mult(ith(i), V[0]->ith(i), Elt1, false);

		for (h = 1; h < nb_with; h++) {
			A->Group_element->element_mult(Elt1, V[h]->ith(i), Elt2, false);
			A->Group_element->element_move(Elt2, Elt1, false);
		}

		A->Group_element->element_move(Elt2, result->ith(i), false);

	}

	FREE_int(Elt1);
	FREE_int(Elt2);

	if (f_v) {
		cout << "vector_ge::multiply_with done" << endl;
	}
}


void vector_ge::conjugate_svas_to(
		int *Elt, vector_ge *&result, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::conjugate_svas_to" << endl;
	}

	result = NEW_OBJECT(vector_ge);
	result->init(
			A, 0 /* verbose_level*/);
	result->allocate(
			len, 0 /* verbose_level*/);
	int i;
	int *Elt1, *Elt2;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->Group_element->invert(Elt, Elt1);

	for (i = 0; i < len; i++) {

		A->Group_element->element_mult(Elt1, ith(i), Elt2, false);
		A->Group_element->element_mult(Elt2, Elt, result->ith(i), false);

	}

	FREE_int(Elt1);
	FREE_int(Elt2);

	if (f_v) {
		cout << "vector_ge::conjugate_svas_to done" << endl;
	}
}


void vector_ge::conjugate_sasv_to(
		int *Elt, vector_ge *&result, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::conjugate_sasv_to" << endl;
	}



	result = NEW_OBJECT(vector_ge);
	result->init(
			A, 0 /* verbose_level*/);
	result->allocate(
			len, 0 /* verbose_level*/);
	int i;
	int *Elt1, *Elt2;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->Group_element->invert(Elt, Elt1);

	for (i = 0; i < len; i++) {

		A->Group_element->element_mult(Elt, ith(i), Elt2, false);
		A->Group_element->element_mult(Elt2, Elt1, result->ith(i), false);

	}

	FREE_int(Elt1);
	FREE_int(Elt2);

	if (f_v) {
		cout << "vector_ge::conjugate_svas_to done" << endl;
	}
}


void vector_ge::field_reduction(
		int subfield_index,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::field_reduction" << endl;
	}


	algebra::basic_algebra::matrix_group *M;

	M = A->get_matrix_group();

	algebra::field_theory::finite_field *F;

	F = M->GFq;

	if (f_v) {
		cout << "vector_ge::field_reduction q=" << F->q << endl;
		cout << "vector_ge::field_reduction subfield_index=" << subfield_index << endl;
	}


	int d;
	int e;

	e = F->e;
	d = M->n * F->e;

	if (f_v) {
		cout << "vector_ge::field_reduction d=" << d << endl;
		cout << "vector_ge::field_reduction e=" << e << endl;
	}

	if (e % subfield_index) {
		cout << "vector_ge::field_reduction "
				"subfield_index must divide field degree" << endl;
		exit(1);
	}

	if (!F->Finite_field_properties->f_related_fields_have_been_computed) {
		cout << "vector_ge::field_reduction "
				"related fields have not yet been computed" << endl;
		exit(1);
	}

	int order_of_subfield;

	algebra::number_theory::number_theory_domain NT;

	int e1;
	int idx;

	e1 = e / subfield_index;

	order_of_subfield = NT.i_power_j(F->p, e1);
	if (f_v) {
		cout << "vector_ge::field_reduction "
				"e1=" << e1 << endl;
		cout << "vector_ge::field_reduction "
				"order_of_subfield=" << order_of_subfield << endl;
	}

	idx = F->Finite_field_properties->Related_fields->position_of_subfield(
			order_of_subfield);
	if (f_v) {
		cout << "vector_ge::field_reduction idx=" << idx << endl;
	}

	//algebra::field_theory::finite_field *Subfield = F->Related_fields->Subfield + idx; // [nb_subfields]

	algebra::field_theory::subfield_structure *SubS =
			F->Finite_field_properties->Related_fields->SubS + idx; // [nb_subfields]

	int h;
	int *Elt;

	for (h = 0; h < len; h++) {

		Elt = ith(h);


		if (M->f_affine) {

			int *B;
			int output_size;


			output_size = d * d + d;


			B = NEW_int(output_size);

			int frob;

			if (M->f_semilinear) {
				frob = Elt[M->n * M->n + M->n];
			}
			else {
				frob = 0;
			}


			SubS->lift_matrix_semilinear(
					Elt /* int *MQ */, frob,
					M->n, B,
					verbose_level - 2);

			// input is MQ[m * m] over the field FQ.
			// output is Mq[n * n] over the field Fq,

			SubS->lift_vector(
					Elt + M->n * M->n,
					M->n, B + d * d, 0 /*verbose_level */);

			// input is vQ[len] over the field FQ.
			// output is vq[len * s] over the field Fq,

			if (f_v) {
				cout << "generator:" << endl;
				Int_matrix_print(Elt, M->n, M->n);
				Int_vec_print(cout, Elt + M->n * M->n, M->n);
				cout << "frob = " << frob << endl;
				cout << "after lifting:" << endl;
				Int_matrix_print(B, d, d);
				Int_vec_print(cout, B + d * d, d);
				cout << endl;
				Int_vec_print(cout, B, output_size);
				cout << endl;
			}

			FREE_int(B);


		}
		else {

			// not affine:

			int *B;
			int output_size;


			output_size = d * d;


			B = NEW_int(output_size);

			int frob;

			if (M->f_semilinear) {
				frob = Elt[M->n * M->n];
			}
			else {
				frob = 0;
			}


			SubS->lift_matrix_semilinear(
					Elt /* int *MQ */, frob,
					M->n, B,
					verbose_level - 2);

			// input is MQ[m * m] over the field FQ.
			// output is Mq[n * n] over the field Fq,

			if (f_v) {
				cout << "generator:" << endl;
				Int_matrix_print(Elt, M->n, M->n);
				cout << "frob = " << frob << endl;
				cout << "after lifting:" << endl;
				Int_matrix_print(B, d, d);
				Int_vec_print(cout, B, output_size);
			}

			FREE_int(B);

		}

	}



	if (f_v) {
		cout << "vector_ge::field_reduction done" << endl;
	}
}


void vector_ge::rational_normal_form(
		vector_ge *&Rational_normal_forms,
		vector_ge *&Base_changes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "vector_ge::rational_normal_form" << endl;
	}

	actions::action_global AGlobal;
	int *Basis;
	int *Rational_normal_form;


	algebra::basic_algebra::matrix_group *M;

	M = A->G.matrix_grp;

	int n;

	n = M->n;

	Rational_normal_forms = NEW_OBJECT(vector_ge);
	Rational_normal_forms->init(A, 0 /* verbose_level */);
	Rational_normal_forms->allocate(len, 0 /* verbose_level */);

	Base_changes = NEW_OBJECT(vector_ge);
	Base_changes->init(A, 0 /* verbose_level */);
	Base_changes->allocate(len, 0 /* verbose_level */);

	Basis = NEW_int(n * n);
	Rational_normal_form = NEW_int(n * n);

	int h;
	int *Elt;

	for (h = 0; h < len; h++) {

		Elt = ith(h);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"h=" << h << " / " << len << endl;
		}

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AGlobal.rational_normal_form" << endl;
		}
		AGlobal.rational_normal_form(
				A,
				Elt,
				Basis,
				Rational_normal_form,
				verbose_level - 2);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AGlobal.rational_normal_form" << endl;
		}
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"h=" << h << " / " << len << " Rational_normal_form=" << endl;
			Int_matrix_print(Rational_normal_form, n, n);
		}
		A->Group_element->make_element(Rational_normal_forms->ith(h), Rational_normal_form, 0 /* verbose_level */);
		A->Group_element->make_element(Base_changes->ith(h), Basis, 0 /* verbose_level */);


	}

	FREE_int(Basis);
	FREE_int(Rational_normal_form);


	if (f_v) {
		cout << "vector_ge::rational_normal_form done" << endl;
	}
}


}}}



