// statistics.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


namespace orbiter {
namespace foundations {


// #############################################################################
// classify.C:
// #############################################################################


//! a statistical analysis of vectors of ints



class classify {

public:
	
	int data_length;
	
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
	
	classify();
	~classify();
	void init(int *data, int data_length, 
		int f_second, int verbose_level);
	int class_of(int pt_idx);
	void print(int f_backwards);
	void print_first(int f_backwards);
	void print_second(int f_backwards);
	void print_file(ostream &ost, int f_backwards);
	void print_file_tex(ostream &ost, int f_backwards);
	void print_naked(int f_backwards);
	void print_naked_tex(ostream &ost, int f_backwards);
	void print_types_naked_tex(ostream &ost, int f_backwards, 
		int *the_vec_sorted, 
		int nb_types, int *type_first, int *type_len);
	double average();
	double average_of_non_zero_values();
	void get_data_by_multiplicity(int *&Pts, int &nb_pts, 
		int multiplicity, int verbose_level);
	int determine_class_by_value(int value);
	void get_class_by_value(int *&Pts, int &nb_pts, int value, 
		int verbose_level);
	set_of_sets *get_set_partition_and_types(int *&types, 
		int &nb_types, int verbose_level);
};


}
}
