// statistics.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

// #############################################################################
// classify.C:
// #############################################################################

class classify {

public:
	
	INT data_length;
	
	INT *data;
	INT *data_sorted;
	INT *sorting_perm;
		// computed using INT_vec_sorting_permutation
	INT *sorting_perm_inv;
		// perm_inv[i] is the index in data 
		// of the element in data_sorted[i]
	INT nb_types;
	INT *type_first;
	INT *type_len;
	
	INT f_second;
	INT *second_data_sorted;
	INT *second_sorting_perm;
	INT *second_sorting_perm_inv;
	INT second_nb_types;
	INT *second_type_first;
	INT *second_type_len;
	
	classify();
	~classify();
	void init(INT *data, INT data_length, 
		INT f_second, INT verbose_level);
	INT class_of(INT pt_idx);
	void print(INT f_backwards);
	void print_first(INT f_backwards);
	void print_second(INT f_backwards);
	void print_file(ostream &ost, INT f_backwards);
	void print_file_tex(ostream &ost, INT f_backwards);
	void print_naked(INT f_backwards);
	void print_naked_tex(ostream &ost, INT f_backwards);
	void print_types_naked_tex(ostream &ost, INT f_backwards, 
		INT *the_vec_sorted, 
		INT nb_types, INT *type_first, INT *type_len);
	double average();
	double average_of_non_zero_values();
	void get_data_by_multiplicity(INT *&Pts, INT &nb_pts, 
		INT multiplicity, INT verbose_level);
	void get_class_by_value(INT *&Pts, INT &nb_pts, INT value, 
		INT verbose_level);
	set_of_sets *get_set_partition_and_types(INT *&types, 
		INT &nb_types, INT verbose_level);
};


