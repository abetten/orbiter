// linear_group_description.cpp
//
// Anton Betten
// December 25, 2015

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {


linear_group_description::linear_group_description()
{
	f_projective = FALSE;
	f_general = FALSE;
	f_affine = FALSE;
	f_GL_d_q_wr_Sym_n = FALSE;
	f_orthogonal = FALSE;
	f_orthogonal_p = FALSE;
	f_orthogonal_m = FALSE;
	GL_wreath_Sym_d = 0;
	GL_wreath_Sym_n = 0;

	f_n = FALSE;
	n = 0;

	//input_q;

	F = NULL;

	f_semilinear = FALSE;
	f_special = FALSE;
	

	// induced actions and subgroups:
	f_wedge_action = FALSE;
	f_wedge_action_detached = FALSE;
	f_PGL2OnConic = FALSE;
	f_monomial_group = FALSE;
	f_diagonal_group = FALSE;
	f_null_polarity_group = FALSE;
	f_symplectic_group = FALSE;
	f_singer_group = FALSE;
	f_singer_group_and_frobenius = FALSE;
	singer_power = 1;
	f_subfield_structure_action = FALSE;
	s = 1;
	f_subgroup_from_file = FALSE;
	f_borel_subgroup_upper = FALSE;
	f_borel_subgroup_lower = FALSE;
	f_identity_group = FALSE;
	//subgroup_fname;
	//subgroup_label;
	f_orthogonal_group = FALSE;
	orthogonal_group_epsilon = 0;

	//f_on_k_subspaces = FALSE;
	//on_k_subspaces_k = 0;

	f_on_tensors = FALSE;
	f_on_rank_one_tensors = FALSE;

	f_subgroup_by_generators = FALSE;
	//subgroup_order_text;
	nb_subgroup_generators = 0;
	//subgroup_generators_label;

	f_Janko1 = FALSE;

	//f_restricted_action = FALSE;
	////restricted_action_text;

	f_export_magma = FALSE;

	f_import_group_of_plane = FALSE;
	//std::string import_group_of_plane_label;

}

linear_group_description::~linear_group_description()
{
}

int linear_group_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level > 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "linear_group_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {


		// the general linear groups:
		// GL, GGL, SL, SSL
		if (ST.stringcmp(argv[i], "-GL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = FALSE;
			f_general = TRUE;
			f_affine = FALSE;
			f_semilinear = FALSE;
			f_special = FALSE;
			if (f_v) {
				cout << "-GL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-GGL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = FALSE;
			f_general = TRUE;
			f_affine = FALSE;
			f_semilinear = TRUE;
			f_special = FALSE;
			if (f_v) {
				cout << "-GGL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-SL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = FALSE;
			f_general = TRUE;
			f_affine = FALSE;
			f_semilinear = FALSE;
			f_special = TRUE;
			if (f_v) {
				cout << "-SL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-SSL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = FALSE;
			f_general = TRUE;
			f_affine = FALSE;
			f_semilinear = TRUE;
			f_special = TRUE;
			if (f_v) {
				cout << "-SSL " << n << " " << input_q << endl;
			}
		}


		// the projective linear groups:
		// PGL, PGGL, PSL, PSSL
		else if (ST.stringcmp(argv[i], "-PGL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = TRUE;
			f_general = FALSE;
			f_affine = FALSE;
			f_semilinear = FALSE;
			f_special = FALSE;
			if (f_v) {
				cout << "-PGL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGGL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = TRUE;
			f_general = FALSE;
			f_affine = FALSE;
			f_semilinear = TRUE;
			f_special = FALSE;
			if (f_v) {
				cout << "-PGGL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PSL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = TRUE;
			f_general = FALSE;
			f_affine = FALSE;
			f_semilinear = FALSE;
			f_special = TRUE;
			if (f_v) {
				cout << "-PSL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PSSL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = TRUE;
			f_general = FALSE;
			f_affine = FALSE;
			f_semilinear = TRUE;
			f_special = TRUE;
			if (f_v) {
				cout << "-PSSL " << n << " " << input_q << endl;
			}
		}



		// the affine groups:
		// AGL, AGGL, ASL, ASSL
		else if (ST.stringcmp(argv[i], "-AGL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = FALSE;
			f_general = FALSE;
			f_affine = TRUE;
			f_semilinear = FALSE;
			f_special = FALSE;
			if (f_v) {
				cout << "-AGL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-AGGL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = FALSE;
			f_general = FALSE;
			f_affine = TRUE;
			f_semilinear = TRUE;
			f_special = FALSE;
			if (f_v) {
				cout << "-AGGL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ASL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = FALSE;
			f_general = FALSE;
			f_affine = TRUE;
			f_semilinear = FALSE;
			f_special = TRUE;
			if (f_v) {
				cout << "-ASL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ASSL") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = FALSE;
			f_general = FALSE;
			f_affine = TRUE;
			f_semilinear = TRUE;
			f_special = TRUE;
			if (f_v) {
				cout << "-ASSL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-GL_d_q_wr_Sym_n") == 0) {
			f_GL_d_q_wr_Sym_n = TRUE;
			GL_wreath_Sym_d = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			GL_wreath_Sym_n = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-GL_d_q_wr_Sym_n " << GL_wreath_Sym_d
					<< " " << input_q << " " << GL_wreath_Sym_n << endl;
			}
		}

		// the orthogonal groups:
		// PGO0, PGOp, PGOm
		else if (ST.stringcmp(argv[i], "-PGO") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_orthogonal = TRUE;
			f_semilinear = FALSE;
			if (f_v) {
				cout << "-PGO " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGOp") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_orthogonal_p = TRUE;
			f_semilinear = FALSE;
			if (f_v) {
				cout << "-PGOp " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGOm") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_orthogonal_m = TRUE;
			f_semilinear = FALSE;
			if (f_v) {
				cout << "-PGOm " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGGO") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_orthogonal = TRUE;
			f_semilinear = TRUE;
			if (f_v) {
				cout << "-PGGO " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGGOp") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_orthogonal_p = TRUE;
			f_semilinear = TRUE;
			if (f_v) {
				cout << "-PGGOp " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGGOm") == 0) {
			f_n = TRUE;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_orthogonal_m = TRUE;
			f_semilinear = TRUE;
			if (f_v) {
				cout << "-PGGOm " << n << " " << input_q << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-wedge") == 0) {
			f_wedge_action = TRUE;
			if (f_v) {
				cout << "-wedge" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-wedge_detached") == 0) {
			f_wedge_action_detached = TRUE;
			if (f_v) {
				cout << "-wedge_detached" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGL2OnConic") == 0) {
			f_PGL2OnConic = TRUE;
			if (f_v) {
				cout << "-PGL2OnConic" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-monomial") == 0) {
			f_monomial_group = TRUE;
			if (f_v) {
				cout << "-monomial " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-diagonal") == 0) {
			f_diagonal_group = TRUE;
			if (f_v) {
				cout << "-diagonal " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-null_polarity_group") == 0) {
			f_null_polarity_group = TRUE;
			if (f_v) {
				cout << "-null_polarity_group" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-symplectic_group") == 0) {
			f_symplectic_group = TRUE;
			if (f_v) {
				cout << "-symplectic_group" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-singer") == 0) {
			f_singer_group = TRUE;
			singer_power = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-singer " << singer_power << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-singer_and_frobenius") == 0) {
			f_singer_group_and_frobenius = TRUE;
			singer_power = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-f_singer_group_and_frobenius " << singer_power << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subfield_structure_action") == 0) {
			f_subfield_structure_action = TRUE;
			s = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-subfield_structure_action " << s << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_from_file") == 0) {
			f_subgroup_from_file = TRUE;
			subgroup_fname.assign(argv[++i]);
			subgroup_label.assign(argv[++i]);
			if (f_v) {
				cout << "-subgroup_from_file " << subgroup_fname
					<< " " << subgroup_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-borel_upper") == 0) {
			f_borel_subgroup_upper = TRUE;
			if (f_v) {
				cout << "-borel_upper" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-borel_lower") == 0) {
			f_borel_subgroup_lower = TRUE;
			if (f_v) {
				cout << "-borel_lower" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identity_group") == 0) {
			f_identity_group = TRUE;
			if (f_v) {
				cout << "-identity_group" << endl;
			}
		}
#if 0
		else if (ST.stringcmp(argv[i], "-on_k_subspaces") == 0) {
			f_on_k_subspaces = TRUE;
			on_k_subspaces_k = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-on_k_subspaces " << on_k_subspaces_k << endl;
			}
		}
#endif
		else if (ST.stringcmp(argv[i], "-on_tensors") == 0) {
			f_on_tensors = TRUE;
			if (f_v) {
				cout << "-on_tensors " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_rank_one_tensors") == 0) {
			f_on_rank_one_tensors = TRUE;
			if (f_v) {
				cout << "-on_rank_one_tensors " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal_group = TRUE;
			orthogonal_group_epsilon = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-orthogonal " << orthogonal_group_epsilon << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-O") == 0) {
			f_orthogonal_group = TRUE;
			orthogonal_group_epsilon = 0;
			if (f_v) {
				cout << "-O" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-O+") == 0 ||
				ST.stringcmp(argv[i], "-Oplus") == 0) {
			f_orthogonal_group = TRUE;
			orthogonal_group_epsilon = 1;
			if (f_v) {
				cout << "-O+" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-O-") == 0 ||
				ST.stringcmp(argv[i], "-Ominus") == 0) {
			f_orthogonal_group = TRUE;
			orthogonal_group_epsilon = -1;
			if (f_v) {
				cout << "-O-" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_by_generators") == 0) {
			f_subgroup_by_generators = TRUE;
			subgroup_label.assign(argv[++i]);
			subgroup_order_text.assign(argv[++i]);
			nb_subgroup_generators = ST.strtoi(argv[++i]);
			subgroup_generators_label.assign(argv[++i]);

			if (f_v) {
				cout << "-subgroup_by_generators " << subgroup_label
						<< " " << nb_subgroup_generators
						<< " " << subgroup_generators_label
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Janko1") == 0) {
			f_Janko1 = TRUE;
			if (f_v) {
				cout << "-Janko1" << endl;
			}
		}
#if 0
		else if (ST.stringcmp(argv[i], "-restricted_action") == 0) {
			f_restricted_action = TRUE;
			restricted_action_text.assign(argv[++i]);
			if (f_v) {
				cout << "-restricted_action " << restricted_action_text << endl;
			}
		}
#endif
		else if (ST.stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			if (f_v) {
				cout << "-export_magma" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-import_group_of_plane") == 0) {
			f_import_group_of_plane = TRUE;
			import_group_of_plane_label.assign(argv[++i]);
			if (f_v) {
				cout << "-import_group_of_plane " << import_group_of_plane_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "linear_group_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "linear_group_description::read_arguments done" << endl;
	}
	return i + 1;
}

void linear_group_description::print()
{
	if (f_import_group_of_plane) {
		cout << "-import_group_of_plane " << import_group_of_plane_label << endl;
	}
	else {
		// the general linear groups:
		// GL, GGL, SL, SSL
		if (!f_affine && !f_special && !f_projective && !f_semilinear) {
			cout << "-GL " << n << " " << input_q << endl;
		}
		if (!f_affine && !f_special && !f_projective && f_semilinear) {
			cout << "-GGL " << n << " " << input_q << endl;
		}
		if (!f_affine && f_special && !f_projective && !f_semilinear) {
			cout << "-SL " << n << " " << input_q << endl;
		}
		if (!f_affine && f_special && !f_projective && f_semilinear) {
			cout << "-SSL " << n << " " << input_q << endl;
		}



		// the projective linear groups:
		// PGL, PGGL, PSL, PSSL
		if (f_projective && !f_affine && !f_special && !f_semilinear) {
			cout << "-PGL " << n << " " << input_q << endl;
		}
		if (f_projective && !f_affine && !f_special && f_semilinear) {
			cout << "-PGGL " << n << " " << input_q << endl;
		}
		if (f_projective && !f_affine && f_special && !f_semilinear) {
			cout << "-PSL " << n << " " << input_q << endl;
		}
		if (f_projective && !f_affine && f_special && f_semilinear) {
			cout << "-PSSL " << n << " " << input_q << endl;
		}



		// the affine groups:
		// AGL, AGGL, ASL, ASSL
		if (f_affine && f_general && !f_special && !f_semilinear) {
			cout << "-AGL " << n << " " << input_q << endl;
		}
		if (f_affine && f_general && !f_special && f_semilinear) {
			cout << "-AGGL " << n << " " << input_q << endl;
		}
		if (f_affine && f_general && f_special && !f_semilinear) {
			cout << "-ASL " << n << " " << input_q << endl;
		}
		if (f_affine && f_general && f_special && f_semilinear) {
			cout << "-ASSL " << n << " " << input_q << endl;
		}
	#if 0
		if (f_override_polynomial) {
			cout << "-override_polynomial" << override_polynomial << endl;
		}
	#endif
		if (f_GL_d_q_wr_Sym_n) {
			cout << "-GL_d_q_wr_Sym_n " << GL_wreath_Sym_d
					<< " " << input_q << " " << GL_wreath_Sym_n << endl;
		}

		// the orthogonal groups:
		// PGO0, PGOp, PGOm
		if (f_orthogonal) {
			cout << "-PGO " << n << " " << input_q << endl;
		}
		if (f_orthogonal_p) {
			cout << "-PGOp " << n << " " << input_q << endl;
		}
		if (f_orthogonal_m) {
			cout << "-PGOm " << n << " " << input_q << endl;
		}
		if (f_orthogonal && f_semilinear) {
			cout << "-PGGO " << n << " " << input_q << endl;
		}
		if (f_orthogonal_p && f_semilinear) {
			cout << "-PGGOp " << n << " " << input_q << endl;
		}
		if (f_orthogonal_m && f_semilinear) {
			cout << "-PGGOm " << n << " " << input_q << endl;
		}


		if (f_wedge_action) {
			cout << "-wedge" << endl;
		}
		if (f_wedge_action_detached) {
			cout << "-wedge_detached" << endl;
		}
		if (f_PGL2OnConic) {
			cout << "-PGL2OnConic" << endl;
		}
		if (f_monomial_group) {
			cout << "-monomial " << endl;
		}
		if (f_diagonal_group) {
			cout << "-diagonal " << endl;
		}
		if (f_null_polarity_group) {
			cout << "-null_polarity_group" << endl;
		}
		if (f_symplectic_group) {
			cout << "-symplectic_group" << endl;
		}
		if (f_singer_group) {
			cout << "-singer " << singer_power << endl;
		}
		if (f_singer_group_and_frobenius) {
			cout << "-f_singer_group_and_frobenius " << singer_power << endl;
		}
		if (f_subfield_structure_action) {
			cout << "-subfield_structure_action " << s << endl;
		}
		if (f_subgroup_from_file) {
			cout << "-subgroup_from_file " << subgroup_fname
					<< " " << subgroup_label << endl;
		}
		if (f_borel_subgroup_upper) {
			cout << "-borel_upper" << endl;
		}
		if (f_borel_subgroup_lower) {
			cout << "-borel_lower" << endl;
		}
		if (f_identity_group) {
			cout << "-identity_group" << endl;
		}
	#if 0
		if (f_on_k_subspaces) {
			cout << "-on_k_subspaces " << on_k_subspaces_k << endl;
		}
	#endif
		if (f_on_tensors) {
			cout << "-on_tensors " << endl;
		}
		if (f_on_rank_one_tensors) {
			cout << "-on_rank_one_tensors " << endl;
		}
		if (f_orthogonal_group) {
			cout << "-orthogonal " << orthogonal_group_epsilon << endl;
		}
		if (f_orthogonal_group) {
			cout << "-O" << endl;
		}
		if (f_orthogonal_group && orthogonal_group_epsilon == 1) {
			cout << "-O+" << endl;
		}
		if (f_orthogonal_group && orthogonal_group_epsilon == -1) {
			cout << "-O-" << endl;
		}
		if (f_subgroup_by_generators) {
			cout << "-subgroup_by_generators " << subgroup_label
					<< " " << nb_subgroup_generators
					<< " " << subgroup_generators_label
					<< endl;
		}
		if (f_Janko1) {
			cout << "-Janko1" << endl;
		}
	#if 0
		if (f_restricted_action) {
			cout << "-restricted_action " << restricted_action_text << endl;
		}
	#endif

	}
	if (f_export_magma) {
		cout << "-export_magma" << endl;
	}
}


}}}


