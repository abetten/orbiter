// linear_group_description.cpp
//
// Anton Betten
// December 25, 2015

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace group_constructions {


linear_group_description::linear_group_description()
{
	f_projective = false;
	f_general = false;
	f_affine = false;
	f_GL_d_q_wr_Sym_n = false;
	f_orthogonal = false;
	f_orthogonal_p = false;
	f_orthogonal_m = false;
	GL_wreath_Sym_d = 0;
	GL_wreath_Sym_n = 0;

	f_n = false;
	n = 0;

	//input_q;

	F = NULL;

	f_semilinear = false;
	f_special = false;
	

	// induced actions and subgroups:
	f_wedge_action = false;
	f_wedge_action_detached = false;

	f_PGL2OnConic = false;
	f_monomial_group = false;
	f_diagonal_group = false;
	f_null_polarity_group = false;
	f_symplectic_group = false;
	f_singer_group = false;
	f_singer_group_and_frobenius = false;
	singer_power = 1;
	f_subfield_structure_action = false;
	s = 1;
	f_subgroup_from_file = false;
	f_borel_subgroup_upper = false;
	f_borel_subgroup_lower = false;
	f_identity_group = false;
	//subgroup_fname;
	//subgroup_label;
	f_orthogonal_group = false;
	orthogonal_group_epsilon = 0;

	f_on_tensors = false;
	f_on_rank_one_tensors = false;

	f_subgroup_by_generators = false;
	//subgroup_order_text;
	nb_subgroup_generators = 0;
	//subgroup_generators_label;

	f_Janko1 = false;

	f_export_magma = false;

	f_import_group_of_plane = false;
	//std::string import_group_of_plane_label;


	f_lex_least_base = false;

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
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = false;
			f_general = true;
			f_affine = false;
			f_semilinear = false;
			f_special = false;
			if (f_v) {
				cout << "-GL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-GGL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = false;
			f_general = true;
			f_affine = false;
			f_semilinear = true;
			f_special = false;
			if (f_v) {
				cout << "-GGL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-SL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = false;
			f_general = true;
			f_affine = false;
			f_semilinear = false;
			f_special = true;
			if (f_v) {
				cout << "-SL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-SSL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = false;
			f_general = true;
			f_affine = false;
			f_semilinear = true;
			f_special = true;
			if (f_v) {
				cout << "-SSL " << n << " " << input_q << endl;
			}
		}


		// the projective linear groups:
		// PGL, PGGL, PSL, PSSL
		else if (ST.stringcmp(argv[i], "-PGL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = true;
			f_general = false;
			f_affine = false;
			f_semilinear = false;
			f_special = false;
			if (f_v) {
				cout << "-PGL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGGL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = true;
			f_general = false;
			f_affine = false;
			f_semilinear = true;
			f_special = false;
			if (f_v) {
				cout << "-PGGL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PSL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = true;
			f_general = false;
			f_affine = false;
			f_semilinear = false;
			f_special = true;
			if (f_v) {
				cout << "-PSL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PSSL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = true;
			f_general = false;
			f_affine = false;
			f_semilinear = true;
			f_special = true;
			if (f_v) {
				cout << "-PSSL " << n << " " << input_q << endl;
			}
		}



		// the affine groups:
		// AGL, AGGL, ASL, ASSL
		else if (ST.stringcmp(argv[i], "-AGL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = false;
			f_general = false;
			f_affine = true;
			f_semilinear = false;
			f_special = false;
			if (f_v) {
				cout << "-AGL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-AGGL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = false;
			f_general = false;
			f_affine = true;
			f_semilinear = true;
			f_special = false;
			if (f_v) {
				cout << "-AGGL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ASL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = false;
			f_general = false;
			f_affine = true;
			f_semilinear = false;
			f_special = true;
			if (f_v) {
				cout << "-ASL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ASSL") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			f_projective = false;
			f_general = false;
			f_affine = true;
			f_semilinear = true;
			f_special = true;
			if (f_v) {
				cout << "-ASSL " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-GL_d_q_wr_Sym_n") == 0) {
			f_GL_d_q_wr_Sym_n = true;
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
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			//f_projective = true;
			f_orthogonal = true;
			f_semilinear = false;
			if (f_v) {
				cout << "-PGO " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGOp") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			//f_projective = true;
			f_orthogonal_p = true;
			f_semilinear = false;
			if (f_v) {
				cout << "-PGOp " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGOm") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			//f_projective = true;
			f_orthogonal_m = true;
			f_semilinear = false;
			if (f_v) {
				cout << "-PGOm " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGGO") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			//f_projective = true;
			f_orthogonal = true;
			f_semilinear = true;
			if (f_v) {
				cout << "-PGGO " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGGOp") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			//f_projective = true;
			f_orthogonal_p = true;
			f_semilinear = true;
			if (f_v) {
				cout << "-PGGOp " << n << " " << input_q << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGGOm") == 0) {
			f_n = true;
			n = ST.strtoi(argv[++i]);
			input_q.assign(argv[++i]);
			//f_projective = true;
			f_orthogonal_m = true;
			f_semilinear = true;
			if (f_v) {
				cout << "-PGGOm " << n << " " << input_q << endl;
			}
		}


		else if (ST.stringcmp(argv[i], "-wedge") == 0) {
			f_wedge_action = true;
			if (f_v) {
				cout << "-wedge" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-wedge_detached") == 0) {
			f_wedge_action_detached = true;
			if (f_v) {
				cout << "-wedge_detached" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-PGL2OnConic") == 0) {
			f_PGL2OnConic = true;
			if (f_v) {
				cout << "-PGL2OnConic" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-monomial") == 0) {
			f_monomial_group = true;
			if (f_v) {
				cout << "-monomial " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-diagonal") == 0) {
			f_diagonal_group = true;
			if (f_v) {
				cout << "-diagonal " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-null_polarity_group") == 0) {
			f_null_polarity_group = true;
			if (f_v) {
				cout << "-null_polarity_group" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-symplectic_group") == 0) {
			f_symplectic_group = true;
			if (f_v) {
				cout << "-symplectic_group" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-singer") == 0) {
			f_singer_group = true;
			singer_power = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-singer " << singer_power << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-singer_and_frobenius") == 0) {
			f_singer_group_and_frobenius = true;
			singer_power = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-f_singer_group_and_frobenius " << singer_power << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subfield_structure_action") == 0) {
			f_subfield_structure_action = true;
			s = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-subfield_structure_action " << s << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_from_file") == 0) {
			f_subgroup_from_file = true;
			subgroup_fname.assign(argv[++i]);
			subgroup_label.assign(argv[++i]);
			if (f_v) {
				cout << "-subgroup_from_file " << subgroup_fname
					<< " " << subgroup_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-borel_upper") == 0) {
			f_borel_subgroup_upper = true;
			if (f_v) {
				cout << "-borel_upper" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-borel_lower") == 0) {
			f_borel_subgroup_lower = true;
			if (f_v) {
				cout << "-borel_lower" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-identity_group") == 0) {
			f_identity_group = true;
			if (f_v) {
				cout << "-identity_group" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_tensors") == 0) {
			f_on_tensors = true;
			if (f_v) {
				cout << "-on_tensors " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-on_rank_one_tensors") == 0) {
			f_on_rank_one_tensors = true;
			if (f_v) {
				cout << "-on_rank_one_tensors " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal_group = true;
			orthogonal_group_epsilon = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-orthogonal " << orthogonal_group_epsilon << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-O") == 0) {
			f_orthogonal_group = true;
			orthogonal_group_epsilon = 0;
			if (f_v) {
				cout << "-O" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-O+") == 0 ||
				ST.stringcmp(argv[i], "-Oplus") == 0) {
			f_orthogonal_group = true;
			orthogonal_group_epsilon = 1;
			if (f_v) {
				cout << "-O+" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-O-") == 0 ||
				ST.stringcmp(argv[i], "-Ominus") == 0) {
			f_orthogonal_group = true;
			orthogonal_group_epsilon = -1;
			if (f_v) {
				cout << "-O-" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-subgroup_by_generators") == 0) {
			f_subgroup_by_generators = true;
			subgroup_label.assign(argv[++i]);
			subgroup_order_text.assign(argv[++i]);
			nb_subgroup_generators = ST.strtoi(argv[++i]);
			subgroup_generators_label.assign(argv[++i]);

			if (f_v) {
				cout << "-subgroup_by_generators "
						<< " " << subgroup_label
						<< " " << subgroup_order_text
						<< " " << nb_subgroup_generators
						<< " " << subgroup_generators_label
						<< endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-Janko1") == 0) {
			f_Janko1 = true;
			if (f_v) {
				cout << "-Janko1" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = true;
			if (f_v) {
				cout << "-export_magma" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-import_group_of_plane") == 0) {
			f_import_group_of_plane = true;
			import_group_of_plane_label.assign(argv[++i]);
			if (f_v) {
				cout << "-import_group_of_plane " << import_group_of_plane_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-lex_least_base") == 0) {
			f_lex_least_base = true;
			if (f_v) {
				cout << "-lex_least_base" << endl;
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
	if (f_orthogonal && !f_semilinear) {
		cout << "-PGO " << n << " " << input_q << endl;
	}
	if (f_orthogonal_p && !f_semilinear) {
		cout << "-PGOp " << n << " " << input_q << endl;
	}
	if (f_orthogonal_m && !f_semilinear) {
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
		cout << "-subgroup_by_generators "
				<< " " << subgroup_label
				<< " " << subgroup_order_text
				<< " " << nb_subgroup_generators
				<< " " << subgroup_generators_label
				<< endl;
	}
	if (f_Janko1) {
		cout << "-Janko1" << endl;
	}
	if (f_export_magma) {
		cout << "-export_magma" << endl;
	}
	if (f_import_group_of_plane) {
		cout << "-import_group_of_plane " << import_group_of_plane_label << endl;
	}
	if (f_lex_least_base) {
		cout << "-lex_least_base" << endl;
	}
}


}}}


