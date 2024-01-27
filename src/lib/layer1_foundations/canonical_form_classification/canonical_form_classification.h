/*
 * canonical_form_classification.h
 *
 *  Created on: Jan 27, 2024
 *      Author: betten
 */

#ifndef SRC_LIB_LAYER1_FOUNDATIONS_CANONICAL_FORM_CLASSIFICATION_CANONICAL_FORM_CLASSIFICATION_H_
#define SRC_LIB_LAYER1_FOUNDATIONS_CANONICAL_FORM_CLASSIFICATION_CANONICAL_FORM_CLASSIFICATION_H_



namespace orbiter {
namespace layer1_foundations {
namespace canonical_form_classification {


// #############################################################################
// classification_of_objects_description.cpp
// #############################################################################




//! description of a classification of objects using class classification_of_objects



class classification_of_objects_description {

public:

	int f_label;
	std::string label;

	int f_save_classification;
	std::string save_prefix;

	int f_max_TDO_depth;
	int max_TDO_depth;

	int f_classification_prefix;
	std::string classification_prefix;

	int f_save_canonical_labeling;

	int f_save_ago;

	int f_save_transversal;


	classification_of_objects_description();
	~classification_of_objects_description();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};

// #############################################################################
// classification_of_objects_report_options.cpp
// #############################################################################




//! options for the report for a classification of combinatorial objects



class classification_of_objects_report_options {

public:

	int f_prefix;
	std::string prefix;

	int f_export_flag_orbits;

	int f_show_incidence_matrices;

	int f_show_TDO;

	int f_show_TDA;

	int f_export_labels;

	int f_export_group_orbiter;
	int f_export_group_GAP;

	int f_lex_least;
	std::string lex_least_geometry_builder;

	classification_of_objects_report_options();
	~classification_of_objects_report_options();
	int read_arguments(
		int argc, std::string *argv,
		int verbose_level);
	void print();

};


// #############################################################################
// classification_of_objects.cpp
// #############################################################################




//! classification of combinatorial objects using a graph-theoretic approach



class classification_of_objects {

public:

	classification_of_objects_description *Descr;

	int f_projective_space;
	//projective_space_with_action *PA;
	geometry::projective_space *P;

	data_input_stream *IS;


	classify_bitvectors *CB;

	long int *Ago; // [IS->nb_objects_to_test]
	int *F_reject; // [IS->nb_objects_to_test]


	// the classification:

	int nb_orbits; // number of isomorphism types

	int *Idx_transversal; // [nb_orbits]

	long int *Ago_transversal; // [nb_orbits]

	object_with_canonical_form **OWCF_transversal; // [nb_orbits]

	l1_interfaces::nauty_output **NO_transversal; // [nb_orbits]


	data_structures::tally *T_Ago;



	classification_of_objects();
	~classification_of_objects();
	void perform_classification(
			classification_of_objects_description *Descr,
			int f_projective_space,
			geometry::projective_space *P,
			data_input_stream *IS,
			int verbose_level);
	void classify_objects_using_nauty(
		int verbose_level);
	void save_automorphism_group_order(
			int verbose_level);
	void save_transversal(
			int verbose_level);
	void process_any_object(
			object_with_canonical_form *OwCF,
			int input_idx, long int &ago, int &f_reject,
			l1_interfaces::nauty_output *&NO,
			encoded_combinatorial_object *&Enc,
			int verbose_level);
	int process_object(
			object_with_canonical_form *OwCF,
		long int &ago,
		int &iso_idx_if_found,
		l1_interfaces::nauty_output *&NO,
		encoded_combinatorial_object *&Enc,
		int verbose_level);
	// returns f_found, which is true if the object is already in the list
	void report_summary_of_orbits(
			std::ostream &ost, int verbose_level);
	void report_all_isomorphism_types(
			std::ostream &ost, int max_TDO_depth,
			int f_show_incma,
			int verbose_level);
	void report_isomorphism_type(
			std::ostream &ost, int i, int max_TDO_depth,
			int f_show_incma,
			int verbose_level);
	void report_object(
			std::ostream &ost,
			object_with_canonical_form *OwCF,
			int object_idx,
			int max_TDO_depth,
			int f_show_incma,
			int verbose_level);


};


// #############################################################################
// classify_bitvectors.cpp
// #############################################################################

//! classification of 0/1 matrices using canonical forms

class classify_bitvectors {
public:

	int nb_types;
		// the number of isomorphism types

	int rep_len;
		// the number of char we need to store the canonical form of
		// one object


	uchar **Type_data;
		// Type_data[nb_types][rep_len]
		// the canonical form of the i-th representative is
		// Type_data[i][rep_len]
	int *Type_rep;
		// Type_rep[nb_types]
		// Type_rep[i] is the index of the candidate which
		// has been chosen as representative
		// for the i-th isomorphism type
	int *Type_mult;
		// Type_mult[nb_types]
		// Type_mult[i] gives the number of candidates which
		// are isomorphic to the i-th isomorphism class representative
	void **Type_extra_data;
		// Type_extra_data[nb_types]
		// Type_extra_data[i] is a pointer that is stored with the
		// i-th isomorphism class representative

	int N;
		// number of candidates (or objects) that we will test
	int n;
		// number of candidates that we have already tested

	int *type_of;
		// type_of[N]
		// type_of[i] is the isomorphism type of the i-th candidate

	data_structures::tally *C_type_of;
		// the classification of type_of[N]
		// this will be computed in finalize()

	int *perm;
		// the permutation which lists the orbit
		// representative in the order
		// in which they appear in the list of candidates

	classify_bitvectors();
	~classify_bitvectors();
	void init(
			int N, int rep_len, int verbose_level);
	int search(
			uchar *data, int &idx, int verbose_level);
	void search_and_add_if_new(
			uchar *data,
			void *extra_data, int &f_found, int &idx,
			int verbose_level);
	// if f_found is true: idx is where the canonical form was found.
	// if f_found is false: idx is where the new canonical form was added.
	int compare_at(
			uchar *data, int idx);
	void add_at_idx(
			uchar *data,
			void *extra_data, int idx, int verbose_level);
	void finalize(
			int verbose_level);
	void print_reps();
	void print_table();
	void save(
			std::string &prefix,
		void (*encode_function)(void *extra_data,
			long int *&encoding, int &encoding_sz, void *global_data),
		void (*get_group_order_or_NULL)(void *extra_data,
				ring_theory::longinteger_object &go, void *global_data),
		void *global_data,
		int verbose_level);

};




// #############################################################################
// classify_using_canonical_forms.cpp
// #############################################################################

//! classification of objects using canonical forms

class classify_using_canonical_forms {
public:


	int nb_input_objects;


	std::vector<data_structures::bitvector *> B;
	std::vector<void *> Objects;
	std::vector<long int> Ago;
	std::vector<int> input_index;

	std::multimap<uint32_t, int> Hashing;
		// we store the pair (hash, idx)
		// where hash is the hash value of the set and idx is the
		// index in the table Sets where the set is stored.
		//
		// we use a multimap because the hash values are not unique
		// it happens that two sets have the same hash value.
		// map cannot handle that.


	//std::vector<void *> Input_objects;
	//std::vector<int> orbit_rep_of_input_object;

	classify_using_canonical_forms();
	~classify_using_canonical_forms();
	void orderly_test(
			object_with_canonical_form *OwCF,
			int &f_accept, int verbose_level);
	void find_object(
			object_with_canonical_form *OwCF,
			int &f_found, int &idx,
			l1_interfaces::nauty_output *&NO,
			data_structures::bitvector *&Canonical_form,
			int verbose_level);
		// if f_found is true, B[idx] agrees with the given object
	void add_object(
			object_with_canonical_form *OwCF,
			int &f_new_object,
			int verbose_level);

};



// #############################################################################
// data_input_stream_description_element.cpp:
// #############################################################################


//! describes one element in an input stream of combinatorial objects


class data_input_stream_description_element {
public:
	enum data_input_stream_type input_type;
	std::string input_string;
	std::string input_string2;

	// for t_data_input_stream_file_of_designs:
	int input_data1; // N_points
	int input_data2; // b = number of blocks
	int input_data3; // k = block size
	int input_data4; // partition class size

	data_input_stream_description_element();
	~data_input_stream_description_element();
	void print();
	void init_set_of_points(
			std::string &a);
	void init_set_of_lines(
			std::string &a);
	void init_set_of_points_and_lines(
			std::string &a, std::string &b);
	void init_packing(
			std::string &a, int q);
	void init_file_of_points(
			std::string &a);
	void init_file_of_points_csv(
			std::string &a, std::string &b);
	void init_file_of_lines(
			std::string &a);
	void init_file_of_packings(
			std::string &a);
	void init_file_of_packings_through_spread_table(
			std::string &a, std::string &b, int q);
	void init_file_of_designs_through_block_orbits(
			std::string &a, std::string &b, int v, int k);
	void init_file_of_point_set(
			std::string &a);
	void init_file_of_designs(
			std::string &a,
				int N_points, int b, int k, int partition_class_size);
	void init_file_of_incidence_geometries(
			std::string &a,
				int v, int b, int f);
	void init_file_of_incidence_geometries_by_row_ranks(
			std::string &a,
				int v, int b, int r);
	void init_incidence_geometry(
			std::string &a,
				int v, int b, int f);
	void init_incidence_geometry_by_row_ranks(
			std::string &a,
				int v, int b, int r);
	void init_from_parallel_search(
			std::string &fname_mask,
			int nb_cases, std::string &cases_fname);

};


// #############################################################################
// data_input_stream_description.cpp:
// #############################################################################


//! description of input data for classification of geometric objects from the command line


class data_input_stream_description {
public:

	int f_label;
	std::string label_txt;
	std::string label_tex;

	int nb_inputs;

	std::vector<data_input_stream_description_element> Input;

	data_input_stream_description();
	~data_input_stream_description();
	int read_arguments(
			int argc, std::string *argv,
		int verbose_level);
	void print();
	void print_item(
			int i);


};


// #############################################################################
// data_input_stream.cpp:
// #############################################################################


//! input data for classification of geometric objects from the command line


class data_input_stream {
public:

	data_input_stream_description *Descr;

	int nb_objects_to_test;

	std::vector<void *> Objects;

	data_input_stream();
	~data_input_stream();
	void init(
			data_input_stream_description *Descr, int verbose_level);
	int count_number_of_objects_to_test(
		int verbose_level);
	void read_objects(
			int verbose_level);


};


// #############################################################################
// encoded_combinatorial_object.cpp
// #############################################################################

//! encoding of combinatorial object for use with nauty


class encoded_combinatorial_object {

private:
	int *Incma; // [nb_rows * nb_cols]

public:
	int nb_rows0;
	int nb_cols0;

	int nb_flags;
	int nb_rows;
	int nb_cols;
	int *partition; // [canonical_labeling_len]

	int canonical_labeling_len; // = nb_rows + nb_cols

	// the permutation representation
	// will be on invariant set:
	int invariant_set_start;
	int invariant_set_size;

	encoded_combinatorial_object();
	~encoded_combinatorial_object();
	void init_everything(
			int nb_rows, int nb_cols,
			int *Incma, int *partition,
			int verbose_level);
	void init_with_matrix(
			int nb_rows, int nb_cols, int *incma,
			int verbose_level);
	void init_row_and_col_partition(
			int *V, int nb_V, int *B, int nb_B,
			int verbose_level);
	void init(
			int nb_rows, int nb_cols,
			int verbose_level);
	std::string stringify_incma();
	int get_nb_flags();
	int *get_Incma();
	void set_incidence_ij(
			int i, int j);
	int get_incidence_ij(
			int i, int j);
	void set_incidence(
			int a);
	void init_canonical_form(
			encoded_combinatorial_object *Enc,
			l1_interfaces::nauty_output *NO,
			int verbose_level);
	void print_incma();
	void save_incma(
			std::string &fname_base, int verbose_level);
	void print_partition();
	void compute_canonical_incma(
			int *canonical_labeling,
			int *&Incma_out, int verbose_level);
	void compute_canonical_form(
			data_structures::bitvector *&Canonical_form,
			int *canonical_labeling, int verbose_level);
	void incidence_matrix_projective_space_top_left(
			geometry::projective_space *P, int verbose_level);
	void extended_incidence_matrix_projective_space_top_left(
			geometry::projective_space *P, int verbose_level);
	void canonical_form_given_canonical_labeling(
			int *canonical_labeling,
			data_structures::bitvector *&B,
			int verbose_level);
	void latex_set_system_by_columns(
			std::ostream &ost,
			int verbose_level);
	void latex_set_system_by_rows(
			std::ostream &ost,
			int verbose_level);
	void latex_incma(
			std::ostream &ost, int verbose_level);
	void latex_TDA(
			std::ostream &ost,
			int nb_orbits, int *orbit_first, int *orbit_len, int *orbit,
			int verbose_level);
	void compute_labels(
			int nb_orbits, int *orbit_first, int *orbit_len, int *orbit,
			int *&point_labels, int *&block_labels,
			int verbose_level);
	void latex_TDA_with_labels(
			std::ostream &ost,
			int nb_orbits, int *orbit_first, int *orbit_len, int *orbit,
			int verbose_level);
	void latex_canonical_form(
			std::ostream &ost,
			l1_interfaces::nauty_output *NO,
			int verbose_level);
	void apply_canonical_labeling(
			int *&Inc2,
			l1_interfaces::nauty_output *NO);
	void apply_canonical_labeling_and_get_flags(
			int *&Inc2,
			int *&Flags, int &nb_flags_counted,
			l1_interfaces::nauty_output *NO);
	void latex_canonical_form_with_labels(
			std::ostream &ost,
			l1_interfaces::nauty_output *NO,
			std::string *row_labels,
			std::string *col_labels,
			int verbose_level);

};


// #############################################################################
// object_with_canonical_form.cpp
// #############################################################################


//! a combinatorial object for which a canonical form can be computed using Nauty



class object_with_canonical_form {
public:
	geometry::projective_space *P;



	object_with_canonical_form_type type;
		// t_PTS = a multiset of points
		// t_LNS = a set of lines
		// t_PNL = a set of points and a set of lines
		// t_PAC = a packing (i.e. q^2+q+1 sets of lines of size q^2+1)
		// t_INC = incidence geometry
		// t_LS = large set

	std::string input_fname;
	int input_idx;
	int f_has_known_ago;
	long int known_ago;

	std::string set_as_string;

	long int *set;
	int sz;
		// set[sz] is used by t_PTS, t_LNS, t_INC

	// for t_PNL:
	long int *set2;
	int sz2;


		// if t_INC or t_LS
	int v;
	int b;
	int f_partition;
	int *partition; // [v + b], do not free !

		// if t_LS
		int design_k;
		int design_sz;

		// t_PAC = packing, uses SoS
		data_structures::set_of_sets *SoS;
		// SoS is used by t_PAC

		int f_extended_incma;

		data_structures::tally *C;
		// used to determine multiplicities in the set of points

	object_with_canonical_form();
	~object_with_canonical_form();
	void print(
			std::ostream &ost);
	void print_rows(
			std::ostream &ost,
			int f_show_incma, int verbose_level);
	void print_tex_detailed(
			std::ostream &ost,
			int f_show_incma, int verbose_level);
	void print_tex(
			std::ostream &ost, int verbose_level);
	void get_packing_as_set_system(
			long int *&Sets,
			int &nb_sets, int &set_size, int verbose_level);
	void init_point_set(
			long int *set, int sz,
		int verbose_level);
	void init_point_set_from_string(
			std::string &set_text,
			int verbose_level);
	void init_line_set(
			long int *set, int sz,
		int verbose_level);
	void init_line_set_from_string(
			std::string &set_text,
			int verbose_level);
	void init_points_and_lines(
		long int *set, int sz,
		long int *set2, int sz2,
		int verbose_level);
	void init_points_and_lines_from_string(
		std::string &set_text,
		std::string &set2_text,
		int verbose_level);
	void init_packing_from_set(
		long int *packing, int sz,
		int verbose_level);
	void init_packing_from_string(
			std::string &packing_text,
			int q,
			int verbose_level);
	void init_packing_from_set_of_sets(
			data_structures::set_of_sets *SoS,
			int verbose_level);
	void init_packing_from_spread_table(
		long int *data,
		long int *Spread_table, int nb_spreads, int spread_size,
		int q,
		int verbose_level);
	void init_design_from_block_orbits(
			data_structures::set_of_sets *Block_orbits,
			long int *Solution, int width,
			int k,
			int verbose_level);
	void init_incidence_geometry(
		long int *data, int data_sz, int v, int b, int nb_flags,
		int verbose_level);
	void init_incidence_geometry_from_vector(
		std::vector<int> &Flags, int v, int b, int nb_flags,
		int verbose_level);
	void init_incidence_geometry_from_string(
		std::string &data,
		int v, int b, int nb_flags,
		int verbose_level);
	void init_incidence_geometry_from_string_of_row_ranks(
		std::string &data,
		int v, int b, int r,
		int verbose_level);
	void init_large_set(
		long int *data, int data_sz, int v, int b, int k, int design_sz,
		int verbose_level);
	void init_large_set_from_string(
		std::string &data_text, int v, int k, int design_sz,
		int verbose_level);
	void encoding_size(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_point_set(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_line_set(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_points_and_lines(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_packing(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_large_set(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void encoding_size_incidence_geometry(
			int &nb_rows, int &nb_cols,
			int verbose_level);
	void canonical_form_given_canonical_labeling(
			int *canonical_labeling,
			data_structures::bitvector *&B,
			int verbose_level);
	void encode_incma(
			encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_point_set(
			encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_line_set(
			encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_points_and_lines(
			encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_packing(
			encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_large_set(
			encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_incidence_geometry(
			encoded_combinatorial_object *&Enc,
			int verbose_level);
	void encode_incma_and_make_decomposition(
			encoded_combinatorial_object *&Enc,
			geometry::incidence_structure *&Inc,
			data_structures::partitionstack *&Stack,
			int verbose_level);
	void encode_object(
			long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_points(
			long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_lines(
			long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_points_and_lines(
			long int *&encoding, int &encoding_sz,
			int verbose_level);
	void encode_object_packing(
			long int *&encoding, int &encoding_sz,
		int verbose_level);
	void encode_object_incidence_geometry(
			long int *&encoding, int &encoding_sz, int verbose_level);
	void encode_object_large_set(
			long int *&encoding, int &encoding_sz, int verbose_level);
	void run_nauty(
			int f_compute_canonical_form,
			data_structures::bitvector *&Canonical_form,
			l1_interfaces::nauty_output *&NO,
			encoded_combinatorial_object *&Enc,
			int verbose_level);
	void canonical_labeling(
			l1_interfaces::nauty_output *NO,
			int verbose_level);
	void run_nauty_basic(
			l1_interfaces::nauty_output *&NO,
			int verbose_level);

};





}}}



#endif /* SRC_LIB_LAYER1_FOUNDATIONS_CANONICAL_FORM_CLASSIFICATION_CANONICAL_FORM_CLASSIFICATION_H_ */
