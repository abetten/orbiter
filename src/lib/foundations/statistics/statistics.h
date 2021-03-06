// statistics.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_STATISTICS_STATISTICS_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_STATISTICS_STATISTICS_H_


namespace orbiter {
namespace foundations {


// #############################################################################
// tally.cpp
// #############################################################################


//! a statistical analysis of data consisting of single integers



class tally {

public:
	
	int data_length;
	
	int f_data_ownership;
	int *data;
	int *data_sorted;
	int *sorting_perm;
		// computed using int_vec_sorting_permutation
	int *sorting_perm_inv;
		// perm_inv[i] is the index in data 
		// of the element in data_sorted[i]
	int nb_types;
	int *type_first;
	int *type_len;
	
	int f_second;
	int *second_data_sorted;
	int *second_sorting_perm;
	int *second_sorting_perm_inv;
	int second_nb_types;
	int *second_type_first;
	int *second_type_len;
	
	tally();
	~tally();
	void init(int *data, int data_length, 
		int f_second, int verbose_level);
	void init_lint(long int *data, int data_length,
		int f_second, int verbose_level);
	void sort_and_classify();
	void sort_and_classify_second();
	int class_of(int pt_idx);
	void print(int f_backwards);
	void print_no_lf(int f_backwards);
	void print_tex_no_lf(int f_backwards);
	void print_first(int f_backwards);
	void print_second(int f_backwards);
	void print_first_tex(int f_backwards);
	void print_second_tex(int f_backwards);
	void print_file(std::ostream &ost, int f_backwards);
	void print_file_tex(std::ostream &ost, int f_backwards);
	void print_file_tex_we_are_in_math_mode(std::ostream &ost, int f_backwards);
	void print_naked_stringstream(std::stringstream &sstr, int f_backwards);
	void print_naked(int f_backwards);
	void print_naked_tex(std::ostream &ost, int f_backwards);
	void print_types_naked_tex(std::ostream &ost, int f_backwards,
		int *the_vec_sorted, 
		int nb_types, int *type_first, int *type_len);
	void print_array_tex(std::ostream &ost, int f_backwards);
	double average();
	double average_of_non_zero_values();
	void get_data_by_multiplicity(int *&Pts, int &nb_pts, 
		int multiplicity, int verbose_level);
	void get_data_by_multiplicity_as_lint(
			long int *&Pts, int &nb_pts, int multiplicity, int verbose_level);
	int determine_class_by_value(int value);
	int get_value_of_class(int class_idx);
	int get_largest_value();
	void get_class_by_value(int *&Pts, int &nb_pts, int value, 
		int verbose_level);
	void get_class_by_value_lint(
			long int *&Pts, int &nb_pts, int value, int verbose_level);
	set_of_sets *get_set_partition_and_types(int *&types, 
		int &nb_types, int verbose_level);
	void save_classes_individually(std::string &fname);
};


// #############################################################################
// tally_vector_data.cpp
// #############################################################################


//! a statistical analysis of data consisting of vectors of ints



class tally_vector_data {

public:

	int data_set_sz;
	int data_length;

	int *data;
	int *rep_idx; // [data_length], rep_idx[i] is the index into Rep of data[i * data_set_sz]
	int *Reps; // [data_length * data_set_sz], used [nb_types * data_set_sz]
	int *Frequency; // [data_length], used [nb_types]
	int *sorting_perm;
		// computed using int_vec_sorting_permutation
	int *sorting_perm_inv;
		// perm_inv[i] is the index in data
		// of the element in data_sorted[i]


	std::multimap<uint32_t, int> Hashing;
		// we store the pair (hash, idx)
		// where hash is the hash value of the set and idx is the
		// index in the table Sets where the set is stored.
		//
		// we use a multimap because the hash values are not unique
		// it happens that two sets have the same hash value.
		// map cannot handle that.

	int nb_types;
	int *type_first;
	//int *type_len; same as Frequency[]

	int **Reps_in_lex_order; // [nb_types]
	int *Frequency_in_lex_order; // [nb_types]


	tally_vector_data();
	~tally_vector_data();
	void init(int *data, int data_length, int data_set_sz,
		int verbose_level);
	int hash_and_find(int *data,
			int &idx, uint32_t &h, int verbose_level);
	void print();
	void save_classes_individually(std::string &fname, int verbose_level);
};


}}

#endif /* ORBITER_SRC_LIB_FOUNDATIONS_STATISTICS_STATISTICS_H_ */


