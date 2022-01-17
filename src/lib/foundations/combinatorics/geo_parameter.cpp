// geo_parameter.cpp
// Anton Betten
//
// started:  May 6, 2008

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {
namespace combinatorics {


geo_parameter::geo_parameter()
{
	decomposition_type = 0;
	fuse_type = 0;
	v = 0;
	b = 0;

	mode = 0;
	//label;

	nb_V = 0;
	nb_B = 0;
	V = NULL;
	B = NULL;
	scheme = NULL;
	fuse = NULL;

	nb_parts = 0;
	nb_entries = 0;

	part = NULL;
	entries = NULL;
	part_nb_alloc = 0;
	entries_nb_alloc = 0;

	lambda_level = 0;
	row_level = 0;
	col_level = 0;
	extra_row_level = 0;
	extra_col_level = 0;
}

geo_parameter::~geo_parameter()
{
	if (V) {
		FREE_int(V);
		V = NULL;
	}
	if (B) {
		FREE_int(B);
		B = NULL;
	}
	if (scheme) {
		FREE_int(scheme);
		scheme = NULL;
	}
	if (fuse) {
		FREE_int(fuse);
		fuse = NULL;
	}
	if (part) {
		FREE_int(part);
		part = NULL;
	}
	if (entries) {
		FREE_int(entries);
		entries = NULL;
	}
}

void geo_parameter::append_to_part(int a)
{
	int nb_alloc;
	
	if (part == NULL) {
		nb_alloc = 1000;
		part = NEW_int(nb_alloc);
		part_nb_alloc = nb_alloc;
		nb_parts = 0;
	}
	else if (nb_parts == part_nb_alloc) {
		nb_alloc = nb_parts + 1000;
		int *new_part, i;
		new_part = NEW_int(nb_alloc);
		for (i = 0; i < nb_parts; i++) {
			new_part[i] = part[i];
		}
		FREE_int(part);
		part = new_part;
		part_nb_alloc = nb_alloc;
	}
	part[nb_parts++] = a;
}

void geo_parameter::append_to_entries(int a1, int a2, int a3, int a4)
{
	int nb_alloc;
	
	if (entries == NULL) {
		nb_alloc = 1000;
		entries = NEW_int(4 * nb_alloc);
		entries_nb_alloc = nb_alloc;
		nb_entries = 0;
	}
	else if (nb_entries == entries_nb_alloc) {
		nb_alloc = nb_entries + 1000;
		int *new_entries, i;
		new_entries = NEW_int(4 * nb_alloc);
		for (i = 0; i < 4 * nb_entries; i++) {
			new_entries[i] = entries[i];
		}
		FREE_int(entries);
		entries = new_entries;
		entries_nb_alloc = nb_alloc;
	}
	entries[4 * nb_entries + 0] = a1;
	entries[4 * nb_entries + 1] = a2;
	entries[4 * nb_entries + 2] = a3;
	entries[4 * nb_entries + 3] = a4;
	nb_entries++;
}

void geo_parameter::write(ofstream &aStream, std::string &label)
{
	
	if (mode == MODE_SINGLE) {
		write_mode_single(aStream, label);
	}
	else if (mode == MODE_STACK) {
		write_mode_stack(aStream, label);
	}
	else {
		cout << "geo_parameter::write unknown mode" << endl;
		exit(1);
	}
}

void geo_parameter::write_mode_single(ofstream &aStream, std::string &label)
{
	int i, j, sum, pt_level = -1, bt_level = -1, xy, x, y, z, w;

	aStream << label << " " << v + b << " " << v << " ";
	sum = V[0];
	for (i = 1; i < nb_V; i++) {
		aStream << sum << " ";
		sum += V[i];
	}
	sum = B[0];
	for (j = 1; j < nb_B; j++) {
		aStream << v + sum << " ";
		sum += B[j];
	}
	aStream << " -1 ";
	if (decomposition_type == POINTTACTICAL ||
			decomposition_type == POINTANDBLOCKTACTICAL) {
		pt_level = nb_V + nb_B;
		for (i = 0; i < nb_V; i++) {
			for (j = 0; j < nb_B; j++) {
				aStream << pt_level << " " 
					<< partition_number_row(i) << " " 
					<< partition_number_col(j) << " " 
					<< scheme[i * nb_B + j] << " ";
			}
		}
	}
	if (decomposition_type == BLOCKTACTICAL ||
			decomposition_type == POINTANDBLOCKTACTICAL) {
		bt_level = nb_V + nb_B;
		for (i = 0; i < nb_V; i++) {
			for (j = 0; j < nb_B; j++) {
				if (decomposition_type == BLOCKTACTICAL) {
					w = scheme[i * nb_B + j];
				}
				else {
					x = scheme[i * nb_B + j];
					y = V[i];
					z = B[j];
					xy = x * y;
					if (xy % z) {
						cout << "geo_parameter::write_mode_single scheme "
								"cannot be block tactical in position "
								<< i << "," << j << endl;
						exit(1);
					}
					w = xy / z;
				}
				aStream << bt_level << " " 
					<< partition_number_col(j) << " " 
					<< partition_number_row(i) << " " 
					<< w << " ";
			}
		}
	}
	aStream << " -1 ";
	int lambda_level = 2;
	int extra_row_level = -1;
	int extra_col_level = -1;
	aStream << pt_level << " " << bt_level << " " << lambda_level << " " << 
		extra_row_level << " " << extra_col_level << endl;
}

void geo_parameter::write_mode_stack(ofstream &aStream, std::string &label)
{
	int i;
	
	aStream << label << " ";
	for (i = 0; i < nb_parts; i++) {
		aStream << part[i] << " ";
	}
	aStream << " -1 ";
	for (i = 0; i < 4 * nb_entries; i++) {
		aStream << entries[i] << " ";
	}
	aStream << " -1 ";
	aStream << row_level << " " 
		<< col_level << " " 
		<< lambda_level << " " 
		<< extra_row_level << " " 
		<< extra_col_level << endl;
	
}

void geo_parameter::convert_single_to_stack(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, sum, x, y, z, w, xy;
	
	row_level = -1;
	col_level = -1;
	lambda_level = 2;
	extra_row_level = -1;
	extra_col_level = -1;
	

	if (f_v) {
		cout << "geo_parameter::convert_single_to_stack" << endl;
		cout << "v=" << v << endl;
		cout << "b=" << b << endl;
	}
	nb_parts = 0;
	nb_entries = 0;
	append_to_part(v + b);
	append_to_part(v);
	sum = V[0];
	for (i = 1; i < nb_V; i++) {
		append_to_part(sum);
		sum += V[i];
	}
	sum = B[0];
	for (j = 1; j < nb_B; j++) {
		append_to_part(v + sum);
		sum += B[j];
	}
	if (f_v) {
		for (i = 0; i < nb_V + nb_B; i++) {
			cout << part[i] << " ";
		}
		cout << endl;
	}
	if (f_v) {
		cout << "decomposition_type=";
		if (decomposition_type == POINTTACTICAL) {
			cout << "POINTTACTICAL" << endl;
		}
		else if (decomposition_type == BLOCKTACTICAL) {
			cout << "BLOCKTACTICAL" << endl;
		}
		else if (decomposition_type == POINTANDBLOCKTACTICAL) {
			cout << "POINTANDBLOCKTACTICAL" << endl;
		}
		
	}
	if (decomposition_type == POINTTACTICAL ||
			decomposition_type == POINTANDBLOCKTACTICAL) {
		row_level = nb_V + nb_B;
		for (i = 0; i < nb_V; i++) {
			for (j = 0; j < nb_B; j++) {
				append_to_entries(row_level, 
					partition_number_row(i),
					partition_number_col(j), 
					scheme[i * nb_B + j]);
				//entries.push_back(row_level);
				//entries.push_back(partition_number_row(i));
				//entries.push_back(partition_number_col(j));
				//entries.push_back(scheme[i * nb_B + j]);
				//nb_entries++;
			}
		}
	}
	if (decomposition_type == BLOCKTACTICAL ||
			decomposition_type == POINTANDBLOCKTACTICAL) {
		col_level = nb_V + nb_B;
		for (i = 0; i < nb_V; i++) {
			for (j = 0; j < nb_B; j++) {
				if (decomposition_type == BLOCKTACTICAL) {
					w = scheme[i * nb_B + j];
				}
				else {
					x = scheme[i * nb_B + j];
					y = V[i];
					z = B[j];
					xy = x * y;
					if (xy % z) {
						cout << "geo_parameter::convert_single_to_stack "
								"scheme cannot be block tactical in "
								"position " << i << "," << j << endl;
						exit(1);
					}
					w = xy / z;
				}
				append_to_entries(col_level, 
					partition_number_col(j), 
					partition_number_row(i), 
					w);
				//entries.push_back(col_level);
				//entries.push_back(partition_number_col(j));
				//entries.push_back(partition_number_row(i));
				//entries.push_back(w);
				//nb_entries++;
			}
		}
	}
	mode = MODE_STACK;


	if (fuse_type == FUSE_TYPE_SIMPLE) {
		//tdo_scheme G;
		
		
		if (decomposition_type == POINTTACTICAL) {
			convert_single_to_stack_fuse_simple_pt(verbose_level);
		}
		else if (decomposition_type == BLOCKTACTICAL) {
			convert_single_to_stack_fuse_simple_bt(verbose_level);
		}
		else {
			cout << "while converting to partition stack, "
					"I cannot figure out which fuse info you meant" << endl;
			exit(1);
		}
	}
	else if (fuse_type == FUSE_TYPE_DOUBLE) {
		//tdo_scheme G;
		
		
		if (decomposition_type == POINTTACTICAL) {
			convert_single_to_stack_fuse_double_pt(verbose_level);
		}
		else if (decomposition_type == BLOCKTACTICAL) {
			cout << "convert_single_to_stack_fuse_double_bt nyi" << endl;
			exit(1);
			//convert_single_to_stack_fuse_double_bt(verbose_level);
		}
		else {
			cout << "while converting to partition stack, "
					"I cannot figure out which fuse info you meant" << endl;
			exit(1);
		}
	}
}

int geo_parameter::partition_number_row(int row_idx)
{
	if (row_idx == 0) {
		return 0;
	}
	else {
		return row_idx + 1;
	}
}
 
int geo_parameter::partition_number_col(int col_idx)
{
	if (col_idx == 0) {
		return 1;
	}
	else {
		return nb_V + col_idx;
	}
}
 
int geo_parameter::input(ifstream &aStream)
{
#ifdef SYSTEMUNIX
	cout << "geo_parameter::input SYSTEMUNIX" << endl;
	string str;
	//Reset();
	//tTDO::Stream(aStream);
	aStream.ignore(INT_MAX, '<');
	aStream >> str;
	v = 0;
	b = 0;
	if (str == "HTDO") {
		cout << "geo_parameter::input read HTDO, "
				"before input_mode_single" << endl;
		return input_mode_single(aStream);
	}
	else {
		cout << "geo_parameter::input str = " << str << endl;
		return FALSE;
	}
#endif
#ifdef SYSTEMWINDOWS
	cout << "geo_parameter::input SYSTEMWINDOWS" << endl;
	cout << "geo_parameter::input has a problem "
			"under windows"<< endl;
	exit(1);
#endif
	cout << "geo_parameter::input neither SYSTEMUNIX "
			"nor SYSTEMWINDOWS" << endl;
	return FALSE;
}

int geo_parameter::input_mode_single(ifstream &aStream)
{
#ifdef SYSTEMUNIX
	int i, j, l, val, eqpos;
	bool brk;
	string str, mapkey, mapval;
	data_structures::string_tools ST;
	

	mode = MODE_SINGLE;
	brk = false;
	label[0] = 0;
	while (!brk) { // read the header
		aStream >> str;
		if (str.substr(str.size() - 1, 1) == ">") {
			str = str.substr(0, str.size() - 1);
			brk = true;
		}
		eqpos = str.find("=");
		if (eqpos > 0) {
			mapkey = str.substr(0, eqpos);
			mapval = str.substr(eqpos + 1, str.size() - eqpos - 1);
			if (mapkey == "type") {
				if (mapval == "pt")
					decomposition_type = POINTTACTICAL;
				else if (mapval == "bt")
					decomposition_type = BLOCKTACTICAL;
				else if (mapval == "geo")
					decomposition_type = POINTANDBLOCKTACTICAL;
				else
					decomposition_type = UNKNOWNTYPE;
			}
			else if (mapkey == "ptanz" || mapkey == "nb_V") {
				nb_V = ST.str2int(mapval);
			}
			else if (mapkey == "btanz" || mapkey == "nb_B") {
				nb_B = ST.str2int(mapval);
			}
			else if (mapkey == "fuse") {
#if 0
				if (mapval == "tdo")
					fuse_type = FUSE_TYPE_TDO;
				else if (mapval == "multi")
					fuse_type = FUSE_TYPE_MULTI;
#endif
				if (mapval == "simple" || mapval == "single")
					fuse_type = FUSE_TYPE_SIMPLE;
				else if (mapval == "double")
					fuse_type = FUSE_TYPE_DOUBLE;
				else {
					cout << "fuse type not recognized" << endl;
					exit(1);
					//fuse_type = FUSE_TYPE_NONE;
				}
			}
			else if (mapkey == "isotest") {
				//_isotestnecessary=str2bool(mapval);
			}
			else if (mapkey == "defekt") {
				//SetDefekt(str2int(mapval));
			}
			else if (mapkey == "id") {
				l = mapval.size();
				if (l >= 1000 - 1) {
					cout << "geo_parameter::input_mode_single "
							"label too long" << endl;
					exit(1);
				}
				for (i = 0; i < l; i++) {
					label[i] = mapval[i];
				}
				label[l] = 0;
#if 0
				ppos = mapval.find_last_of(".");
				_idadd = mapval.substr(0, ppos + 1);
				mapval = mapval.substr(ppos + 1, mapval.size() - ppos - 1);
				_id = str2int(mapval);
#endif
			}
		}
			brk = brk || aStream.eof();
	}
	//cout << "nb_V=" << nb_V << " nb_B=" << nb_B << endl;
	V = NEW_int(nb_V);
	B = NEW_int(nb_B);
	scheme = NEW_int(nb_V * nb_B);
	fuse = NULL;
	if (decomposition_type == UNKNOWNTYPE) {
		cout << "decomposition_type needs to be defined"<<endl;
		exit(1);
	}
	if (nb_V == 0) {
		cout << "nb_V == 0"<<endl;
		exit(1);
	}
	if (nb_B == 0) {
		cout << "nb_B == 0"<<endl;
		exit(1);
	}
	v = 0;
	b = 0;
	for (j = 0; j < nb_B; j++) {
		//cout << "j=" << j << endl;
		aStream >> val;
		//cout << "read " << val << endl;
		B[j] = val;
		b += val;
	}
	for (i = 0; i < nb_V; i++) {
		aStream >> val;
		//cout << "read " << val << endl;
		V[i] = val;
		v += val;
		for (j = 0; j < nb_B; j++) {
			aStream >> val;
			//cout << "read " << val << endl;
			scheme[i * nb_B + j] = val;
		}
	}
	//cout << "after reading scheme" << endl;
#if 0
	if (fuse_type == FUSE_TYPE_MULTI) {
		for (i = 0; i < nb_V; i++) {
			for (j = 0; j < nb_B; j++) {
				aStream >> val;
				//cout << "read " << val << endl;
				fuse[i * nb_B + j] = val;
			}
		}
	}
#endif
	if (fuse_type == FUSE_TYPE_SIMPLE) {
		if (decomposition_type == POINTTACTICAL) {
			fuse = NEW_int(nb_V);
			for (i = 0; i < nb_V; i++) {
				aStream >> val;
				//cout << "read " << val << endl;
				fuse[i] = val;
			}
		}
		else if (decomposition_type == BLOCKTACTICAL) {
			fuse = NEW_int(nb_B);
			for (i = 0; i < nb_B; i++) {
				aStream >> val;
				//cout << "read " << val << endl;
				fuse[i] = val;
			}
		}
		else {
			cout << "while reading fuse, I cannot figure out "
					"which fuse info you meant" << endl;
			exit(1);
		}
	}
	else if (fuse_type == FUSE_TYPE_DOUBLE) {
		if (decomposition_type == POINTTACTICAL) {
			fuse = NEW_int(4 + 2 * nb_V);
			for (i = 0; i < 4 + 2 * nb_V; i++) {
				aStream >> val;
				//cout << "read " << val << endl;
				fuse[i] = val;
			}
		}
		else if (decomposition_type == BLOCKTACTICAL) {
			fuse = NEW_int(4 + 2 * nb_B);
			for (i = 0; i < 4 + (2 * nb_B); i++) {
				aStream >> val;
				//cout << "read " << val << endl;
				fuse[i] = val;
			}
		}
		else {
			cout << "while reading fuse, I cannot figure out "
					"which fuse info you meant" << endl;
			exit(1);
		}
	}
	//cout << "before ignore" << endl;
	aStream.ignore(INT_MAX, '>');
	//cout << "geo_parameter::input_mode_single
	//v=" << v << " b=" << b << endl;
#endif
#ifdef SYSTEMWINDOWS
	cout << "geo_parameter::input_mode_single "
			"has a problem under windows" << endl;
	exit(1);
#endif
	return TRUE;
}

int geo_parameter::input_mode_stack(
		ifstream &aStream, int verbose_level)
{
#if defined(SYSTEMUNIX) || defined(SYSTEM_IS_MACINTOSH)
	int f_v = (verbose_level >= 1);
	string str;
	int i, l, val, v1, v2, v3, v4;
	
	//part.clear();
	//entries.clear();
	
	if (f_v) {
		cout << "geo_parameter::input_mode_stack" << endl;
	}
	aStream >> str;
	if (str == "-1") {
		return FALSE;
	}
	if (f_v) {
		cout << "geo_parameter::input_mode_stack "
				"read \"" << str << "\"" << endl;
	}
	l = str.length();
	if (l >= 1000 - 1) {
		cout << "geo_parameter::input_mode_stack "
				"label too long" << endl;
		exit(1);
	}
	for (i = 0; i < l; i++) {
		label[i] = str[i];
	}
	label[l] = 0;
	if (f_v) {
		cout << "geo_parameter::input_mode_stack "
				"label='" << label << "'" << endl;
	}
	nb_parts = 0;
	if (part) {
		FREE_int(part);
		part = NULL;
	}
	if (f_v) {
		cout << "geo_parameter::input_mode_stack "
				"reading part" << endl;
	}
	part = NULL;
	while (TRUE) {
		aStream >> val;
		if (val == -1) {
			append_to_part(val);
			nb_parts--;
			//part.push_back(val);
			break;
		}
		append_to_part(val);
	}
	if (f_v) {
		cout << "geo_parameter::input_mode_stack "
				"reading entries" << endl;
	}
	nb_entries = 0;
	if (entries) {
		FREE_int(entries);
		entries = NULL;
	}
	entries = NULL;
	while (TRUE) {
		aStream >> v1;
		if (v1 == -1) {
			break;
		}

		aStream >> v2;

		aStream >> v3;

		aStream >> v4;

		append_to_entries(v1, v2, v3, v4);

	}
	if (f_v) {
		cout << "geo_parameter::input_mode_stack "
				"reading row_level" << endl;
	}
	aStream >> row_level;
	aStream >> col_level;
	aStream >> lambda_level;
	aStream >> extra_row_level;
	aStream >> extra_col_level;
	//cout << "nb_parts=" << nb_parts << endl;
	//cout << "nb_entries=" << nb_entries << endl;
#endif
#ifdef SYSTEMWINDOWS
	cout << "geo_parameter::input_mode_stack has a "
			"problem under windows" << endl;
	exit(1);
#endif
	if (f_v) {
		cout << "geo_parameter::input_mode_stack done" << endl;
	}
	return TRUE;
}

void geo_parameter::init_tdo_scheme(tdo_scheme_synthetic &G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "geo_parameter::init_tdo_scheme" << endl;
		cout << "nb_parts=" << nb_parts << endl;
		cout << "nb_entries=" << nb_entries << endl;
	}
	G.part = NEW_int(nb_parts + 1);
	for (i = 0; i < nb_parts; i++) {
		G.part[i] = part[i];
	}
	G.part[nb_parts] = -1;
	G.part_length = nb_parts;
	G.entries = NEW_int(nb_entries * 4 + 1);
	for (i = 0; i < 4 * nb_entries; i++) {
		G.entries[i] = entries[i];
	}
	G.entries[4 * nb_entries] = -1;
	G.nb_entries = nb_entries;
	
	G.row_level = row_level;
	G.col_level = col_level;
	G.extra_row_level = extra_row_level;
	G.extra_col_level = extra_col_level;
	G.lambda_level = lambda_level;
	
	G.level[ROW_SCHEME] = row_level;
	G.level[COL_SCHEME] = col_level;
	G.level[EXTRA_ROW_SCHEME] = extra_row_level;
	G.level[EXTRA_COL_SCHEME] = extra_col_level;
	G.level[LAMBDA_SCHEME] = lambda_level;

	if (f_v) {
		cout << "geo_parameter::init_tdo_scheme "
				"calling G.init_partition_stack" << endl;
	}
	G.init_partition_stack(verbose_level - 5);
	if (f_v) {
		cout << "after G.init_partition_stack" << endl;
	}
}

void geo_parameter::print_schemes(tdo_scheme_synthetic &G)
{
	cout << "geo_parameter::print_schemes" << endl;
	cout << "decomposition " << label << ":" << endl;
	G.print_scheme(LAMBDA_SCHEME, FALSE);
	if (row_level >= 2) {
		G.print_scheme(ROW_SCHEME, FALSE);
	}
	if (col_level >= 2) {
		G.print_scheme(COL_SCHEME, FALSE);
	}
	if (extra_row_level > 2) {
		G.print_scheme(EXTRA_ROW_SCHEME, FALSE);
	}
	if (extra_col_level > 2) {
		G.print_scheme(EXTRA_COL_SCHEME, FALSE);
	}
}

void geo_parameter::print_schemes_tex(tdo_scheme_synthetic &G)
{
	cout << "decomposition " << label << ":" << endl;
	G.print_scheme_tex_fancy(cout, LAMBDA_SCHEME, TRUE, label);
	if (row_level >= 2) {
		G.print_scheme_tex_fancy(cout, ROW_SCHEME, TRUE, label);
	}
	if (col_level >= 2) {
		G.print_scheme_tex_fancy(cout, COL_SCHEME, TRUE, label);
	}
	if (extra_row_level > 2) {
		G.print_scheme_tex_fancy(cout, EXTRA_ROW_SCHEME, TRUE, label);
	}
	if (extra_col_level > 2) {
		G.print_scheme_tex_fancy(cout, EXTRA_COL_SCHEME, TRUE, label);
	}
}

void geo_parameter::print_scheme_tex(ostream &ost, tdo_scheme_synthetic &G, int h)
{
	G.print_scheme_tex_fancy(ost, h, TRUE, label);
}

void geo_parameter::print_C_source()
{
	int i, j;
	
	cout << "char *name = \"" << label << "\";" << endl;
	cout << "int part[] = {";
	for (i = 0; i < nb_parts; i++) 
		cout << part[i] << ",";
	cout << "-1};" << endl;
	cout << "// this is the partition of rows and columns" << endl;
	cout << "int entries[] = {" << endl;
		for (i = 0; i < nb_entries; i++) {
			for (j = 0; j < 4; j++) {
				cout << setw(3) << entries[i * 4 + j] << ",";
			}
			cout << endl;
		}
		cout << "-1, };" << endl;
	cout << "int col_level = " << col_level << ";" << endl;
	cout << "int row_level = " << row_level << ";" << endl;
	cout << "int extra_row_level = " << extra_row_level << ";" << endl;
	cout << "int extra_col_level = " << extra_col_level << ";" << endl;
	cout << "int lambda_level = " << lambda_level << ";" << endl;
	cout << "int f_has_lambda = TRUE;" << endl;
	cout << "int lambda_target[] = {1};" << endl;
	cout << "int lambda_mode[] = {EQ};" << endl;
	cout << "int f_has_mu = FALSE;" << endl;
	cout << "int mu = 0;" << endl;
	cout << "int f_girth = FALSE;" << endl;
	cout << "int girth = 0;" << endl;
	cout << "int f_starter = FALSE;" << endl;
	cout << "int starter[] = {};" << endl;
	cout << "int listofcases_table_depth = 0;" << endl;
	cout << "int listofcases_table_length = 0;" << endl;
	cout << "int listofcases_table[] = {" << endl;
	cout << "};" << endl;
	cout << "int AGO_THRESHOLD  = 100000000;" << endl;
	cout << "int PRint_intERVAL = 10000;" << endl;
	cout << "int PRint_intERVAL_HOURS = 0;" << endl;
	cout << "int PRint_intERVAL_MINUTES = 0;" << endl;
	cout << "int PRint_intERVAL_SECONDS = 10;" << endl;
}

void geo_parameter::convert_single_to_stack_fuse_simple_pt(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int I, u, l, i, j, a, sum, s, M, c, e, h, c1, c2, f;
	tdo_scheme_synthetic G;

	if (f_v) {
		cout << "geo_parameter::convert_single_to_stack_fuse_simple_pt" << endl;
	}
	

	int *class_first, *class_len, nb_classes;
	int *block_length;
	int *prev_scheme;
	int *class_relabel;
	int prev_level;
			
	if (f_v) {
		cout << "processing fuse simple for "
				"pointtactical decomposition" << endl;
	}
	init_tdo_scheme(G, verbose_level);
	h = ROW_SCHEME;
			
	class_first = NEW_int(nb_V);
	class_len = NEW_int(nb_V);
	block_length = NEW_int(nb_V);
	class_relabel = NEW_int(nb_V + nb_B);
	for (i = 0; i < nb_V + nb_B; i++) {
		class_relabel[i] = -1;
	}
	class_relabel[0] = 0;
	class_relabel[nb_V] = 1;

	int_vec_classify(fuse, nb_V, class_first, class_len, nb_classes);

#if 0
	class_first[0] = 0;
	class_len[0] = 1;
	for (i = 1; i < nb_V; i++) {
		if (fuse[i] == fuse[i - 1]) {
			class_len[nb_classes]++;
			}
		else {
			nb_classes++;
			class_first[nb_classes] =
				class_first[nb_classes - 1] + class_len[nb_classes - 1];
			class_len[nb_classes] = 1;
			}
		}
	nb_classes++;
#endif

	prev_scheme = NEW_int(nb_classes * nb_B);
	for (I = 0; I < nb_classes; I++) {
		block_length[I] = tdo_scheme_get_row_class_length_fused(G, h, 
			class_first[I], class_len[I]);
#if 0
		l = class_len[I];
		L = 0;
		for (u = 0; u < l; u++) {
			i = class_first[I] + u;
			a = G.row_classes_len[h][i];
			L += a;
			}
		block_length[I] = L;
#endif
	}
	if (f_v) {
		cout << "found " << nb_classes << " classes in the "
				"previous row decomposition" << endl;
		cout << "class_first: ";
		Orbiter->Int_vec->print(cout, class_first, nb_classes);
		cout << endl;
		cout << "class_len: ";
		Orbiter->Int_vec->print(cout, class_len, nb_classes);
		cout << endl;
		cout << "block_length: ";
		Orbiter->Int_vec->print(cout, block_length, nb_classes);
		cout << endl;
	}
	
	for (j = 0; j < nb_B; j++) {
		for (I = 0; I < nb_classes; I++) {
			l = class_len[I];
			s = 0;
			for (u = 0; u < l; u++) {
				i = class_first[I] + u;
				a = G.row_classes_len[h][i];
				c = scheme[i * nb_B + j];
				s += a * c;
			}
			M = G.col_classes_len[h][j];
			e = s / M;
			if (e * M != s) {
				cout << "problems figuring out the previous scheme" << endl;
				cout << "s=" << s << endl;
				//cout << "L=" << L << endl;
				cout << "M=" << M << endl;
				cout << "j=" << j << endl;
				cout << "I=" << I << endl;
				exit(1);
			}
			prev_scheme[I * nb_B + j] = e;
		}
	} // next j
	if (f_v) {
		cout << "the previous col scheme is" << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout,
			prev_scheme, nb_classes, nb_B, nb_B, 4);
	}
	//part.clear();
	nb_parts = 0;
	append_to_part(v + b);
	append_to_part(v);
	//part.push_back(v + b);
	//part.push_back(v);
	//nb_parts = 2;
	sum = block_length[0];
	for (I = 1; I < nb_classes; I++) {
		class_relabel[class_first[I]] = nb_parts;
		append_to_part(sum);
		//part.push_back(sum);
		//nb_parts++;
		sum += block_length[I];
	}
	sum = B[0];
	for (j = 1; j < nb_B; j++) {
		class_relabel[nb_V + j] = nb_parts;
		append_to_part(v + sum);
		//part.push_back(v + sum);
		//nb_parts++;
		sum += B[j];
	}
	prev_level = nb_parts;
	if (f_v) {
		cout << "the previous decomposition is " << endl;
		Orbiter->Int_vec->print(cout, part, nb_parts);
		//for (i = 0; i < nb_parts; i++) {
			//cout << part[i] << " ";
			//}
		cout << endl;
		cout << "class_relabel: " << endl;
		Orbiter->Int_vec->print(cout, class_relabel, nb_V + nb_B);
		cout << endl;
	}
	nb_entries = 0;
	//entries.clear();
	for (I = 0; I < nb_classes; I++) {
		for (j = 0; j < nb_B; j++) {
			c = prev_scheme[I * nb_B + j];
			if (I == 0)
				c1 = 0;
			else
				c1 = I + 1;
			if (j == 0)
				c2 = 1;
			else
				c2 = nb_classes + j;	
			if (f_v) {
				cout << "entry " << nb_entries << " : " << prev_level
					<< " " << c2 << " " << c1 << " " << c << endl;
				}
			append_to_entries(prev_level, c2, c1, c);
		}
	}
	nb_parts = prev_level;
	col_level = prev_level;
	sum = 0;
	for (I = 0; I < nb_classes; I++) {
		l = class_len[I];
		f = class_first[I];
		sum += G.row_classes_len[h][f + 0];
		for (u = 1; u < l; u++) {
			class_relabel[f + u] = nb_parts;
			//part.push_back(sum);
			//nb_parts++;
			append_to_part(sum);
			sum += G.row_classes_len[h][f + u];
		}
	}
	if (f_v) {
		cout << "the extended decomposition is " << endl;
		Orbiter->Int_vec->print(cout, part, nb_parts);
		//for (i = 0; i < nb_parts; i++) {
			//cout << part[i] << " ";
			//}
		cout << endl;
		cout << "class_relabel: " << endl;
		Orbiter->Int_vec->print(cout, class_relabel, nb_V + nb_B);
		cout << endl;
	}
	row_level = nb_parts;
	for (i = 0; i < nb_V; i++) {
		for (j = 0; j < nb_B; j++) {
			c = scheme[i * nb_B + j];
			c1 = class_relabel[i];
			c2 = class_relabel[nb_V + j];
			if (f_v) {
				cout << "entry " << nb_entries << " : " << row_level
					<< " " << c1 << " " << c2 << " " << c << endl;
			}
			append_to_entries(row_level, c1, c2, c);
			//nb_entries++;
		}
	}
	FREE_int(class_first);
	FREE_int(class_len);
	FREE_int(block_length);
	FREE_int(prev_scheme);
	FREE_int(class_relabel);
	//exit(1);
}

void geo_parameter::convert_single_to_stack_fuse_simple_bt(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int J, u, l, i, j, a, sum, s, L, M, c, e, h, c1, c2, f;
	tdo_scheme_synthetic G;

	if (f_v) {
		cout << "geo_parameter::convert_single_to_stack_fuse_simple_bt" << endl;
	}
	
	int *class_first, *class_len, nb_classes;
	int *block_length;
	int *prev_scheme;
	int *class_relabel;
	int prev_level;
			
	if (f_v) {
		cout << "processing fuse simple for "
				"blocktactical decomposition" << endl;
	}
	init_tdo_scheme(G, verbose_level);
	h = COL_SCHEME;
	class_first = NEW_int(nb_B);
	class_len = NEW_int(nb_B);
	block_length = NEW_int(nb_B);
	class_relabel = NEW_int(nb_V + nb_B);
	for (i = 0; i < nb_V + nb_B; i++) {
		class_relabel[i] = -1;
	}
	class_relabel[0] = 0;
	class_relabel[nb_V] = 1;

	int_vec_classify(fuse, nb_B, class_first, class_len, nb_classes);

#if 0
	class_first[0] = 0;
	class_len[0] = 1;
	for (i = 1; i < nb_B; i++) {
		if (fuse[i] == fuse[i - 1]) {
			class_len[nb_classes]++;
		}
		else {
			nb_classes++;
			class_first[nb_classes] =
					class_first[nb_classes - 1] +
					class_len[nb_classes - 1];
			class_len[nb_classes] = 1;
		}
	}
	nb_classes++;
#endif
	prev_scheme = NEW_int(nb_V * nb_classes);
	for (J = 0; J < nb_classes; J++) {
		block_length[J] = tdo_scheme_get_col_class_length_fused(G, h, 
			class_first[J], class_len[J]);
#if 0
		l = class_len[J];
		L = 0;
		for (u = 0; u < l; u++) {
			j = class_first[J] + u;
			a = G.col_classes_len[h][j];
			L += a;
		}
		block_length[J] = L;
#endif
	}

	if (f_v) {
		cout << "found " << nb_classes << " classes in the previous "
				"column decomposition" << endl;
		cout << "class_first: ";
		Orbiter->Int_vec->print(cout, class_first, nb_classes);
		cout << endl;
		cout << "class_len: ";
		Orbiter->Int_vec->print(cout, class_len, nb_classes);
		cout << endl;
		cout << "block_length: ";
		Orbiter->Int_vec->print(cout, block_length, nb_classes);
		cout << endl;
	}
	
	for (i = 0; i < nb_V; i++) {
		for (J = 0; J < nb_classes; J++) {
			l = class_len[J];
			s = 0;
			L = 0;
			for (u = 0; u < l; u++) {
				j = class_first[J] + u;
				a = G.col_classes_len[h][j];
				c = scheme[i * nb_B + j];
				s += a * c;
				L += a;
			}

			M = G.row_classes_len[h][i];
			e = s / M;
			if (e * M != s) {
				cout << "problems figuring out the previous scheme" << endl;
				exit(1);
			}
			prev_scheme[i * nb_classes + J] = e;
		}
	} // next j
	if (f_v) {
		cout << "the previous row scheme is" << endl;
		Orbiter->Int_vec->print_integer_matrix_width(cout,
				prev_scheme, nb_V, nb_classes, nb_classes, 4);
	}
	nb_parts = 0;
	//part.clear();
	append_to_part(v + b);
	append_to_part(v);
	//part.push_back(v + b);
	//part.push_back(v);
	//nb_parts = 2;
	sum = block_length[0];
	for (J = 1; J < nb_classes; J++) {
		class_relabel[nb_V + class_first[J]] = nb_parts;
		append_to_part(v + sum);
		//part.push_back(v + sum);
		//nb_parts++;
		sum += block_length[J];
	}
	sum = V[0];
	for (i = 1; i < nb_V; i++) {
		class_relabel[i] = nb_parts;
		append_to_part(sum);
		//nb_parts++;
		sum += V[i];
	}
	prev_level = nb_parts;
	if (f_v) {
		cout << "the previous decomposition is " << endl;
		Orbiter->Int_vec->print(cout, part, nb_parts);
		//for (i = 0; i < nb_parts; i++) {
			//cout << part[i] << " ";
			//}
		cout << endl;
		cout << "class_relabel: " << endl;
		Orbiter->Int_vec->print(cout, class_relabel, nb_V + nb_B);
		cout << endl;
	}
	nb_entries = 0;
	//entries.clear();
	for (i = 0; i < nb_V; i++) {
		for (J = 0; J < nb_classes; J++) {
			c = prev_scheme[i * nb_classes + J];
			if (i == 0) {
				c1 = 0;
			}
			else {
				c1 = nb_classes + i;
			}
			if (J == 0) {
				c2 = 1;
			}
			else {
				c2 = 1 + J;
			}
			if (f_v) {
				cout << "entry " << nb_entries << " : " << prev_level
					<< " " << c1 << " " << c2 << " " << c << endl;
			}
			append_to_entries(prev_level, c1, c2, c);
			//nb_entries++;
		}
	}
	nb_parts = prev_level;
	row_level = prev_level;
	sum = v;
	for (J = 0; J < nb_classes; J++) {
		l = class_len[J];
		f = class_first[J];
		sum += G.col_classes_len[h][f + 0];
		for (u = 1; u < l; u++) {
			class_relabel[nb_V + f + u] = nb_parts;
			append_to_part(sum);
			//part.push_back(sum);
			//nb_parts++;
			sum += G.col_classes_len[h][f + u];
		}
	}
	if (f_v) {
		cout << "the extended decomposition is " << endl;
		Orbiter->Int_vec->print(cout, part, nb_parts);
		//for (i = 0; i < nb_parts; i++) {
			//cout << part[i] << " ";
			//}
		cout << endl;
		cout << "class_relabel: " << endl;
		Orbiter->Int_vec->print(cout, class_relabel, nb_V + nb_B);
		cout << endl;
	}
	col_level = nb_parts;
	for (i = 0; i < nb_V; i++) {
		for (j = 0; j < nb_B; j++) {
			c = scheme[i * nb_B + j];
			c1 = class_relabel[i];
			c2 = class_relabel[nb_V + j];
			if (f_v) {
				cout << "entry " << nb_entries << " : " << col_level
					<< " " << c2 << " " << c1 << " " << c << endl;
			}
			append_to_entries(col_level, c2, c1, c);
			//nb_entries++;
		}
	}
	FREE_int(class_first);
	FREE_int(class_len);
	FREE_int(block_length);
	FREE_int(prev_scheme);
	FREE_int(class_relabel);
	//exit(1);
}


void int_vec_classify(int *v, int len,
		int *class_first, int *class_len, int &nb_classes)
{
	int i;
	
	nb_classes = 0;
	class_first[0] = 0;
	class_len[0] = 1;
	for (i = 1; i < len; i++) {
		if (v[i] == v[i - 1]) {
			class_len[nb_classes]++;
		}
		else {
			nb_classes++;
			class_first[nb_classes] =
					class_first[nb_classes - 1] + class_len[nb_classes - 1];
			class_len[nb_classes] = 1;
		}
	}
	nb_classes++;
}

int tdo_scheme_get_row_class_length_fused(tdo_scheme_synthetic &G,
		int h, int class_first, int class_len)
{
	int L, u;
	
	L = 0;
	for (u = 0; u < class_len; u++) {
		L += G.row_classes_len[h][class_first + u];
	}
	return L;
}

int tdo_scheme_get_col_class_length_fused(tdo_scheme_synthetic &G,
		int h, int class_first, int class_len)
{
	int L, u;
	
	L = 0;
	for (u = 0; u < class_len; u++) {
		L += G.col_classes_len[h][class_first + u];
	}
	return L;
}

void geo_parameter::convert_single_to_stack_fuse_double_pt(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int I, u, l, i, j, ii, jj, a, sum, s, M, c, e, h, c1, c2, f, d;
	int fuse_block_first[2], fuse_block_len[2];
	int *the_fuse[2];
	
	tdo_scheme_synthetic G;

	if (f_v) {
		cout << "geo_parameter::convert_single_to_stack_fuse_double_pt" << endl;
	}
	fuse_block_first[0] = fuse[0];
	fuse_block_len[0] = fuse[1];
	fuse_block_first[1] = fuse[2];
	fuse_block_len[1] = fuse[3];
	the_fuse[0] = fuse + 4;
	the_fuse[1] = the_fuse[0] + nb_V;
	

	int *class_first[2], *class_len[2], *class_idx[3], nb_classes[2];
	int *block_length[2];
	int *prev_scheme[2];
	int *class_relabel;
	int prev_level[2];
			
	if (f_v) {
		cout << "processing fuse double for "
				"pointtactical decomposition" << endl;
		cout << "fuse_block_first[0]=" << fuse_block_first[0] << endl;
		cout << "fuse_block_len[0]  =" << fuse_block_len[0] << endl;
		cout << "fuse_block_first[1]=" << fuse_block_first[1] << endl;
		cout << "fuse_block_len[1]  =" << fuse_block_len[1] << endl;
		cout << "the_fuse[0] : ";
		Orbiter->Int_vec->print(cout, the_fuse[0], nb_V);
		cout << endl;
		cout << "the_fuse[1] : ";
		Orbiter->Int_vec->print(cout, the_fuse[1], nb_V);
		cout << endl;
	}
	init_tdo_scheme(G, 0 /*verbose_level*/);
	h = ROW_SCHEME;
			
	class_relabel = NEW_int(nb_V + nb_B);
	for (i = 0; i < nb_V + nb_B; i++) {
		class_relabel[i] = -1;
	}
	class_relabel[0] = 0;
	class_relabel[nb_V] = 1;


	
	
	for (d = 0; d < 2; d++) {
		if (f_vv) {
			cout << "d=" << d << endl;
		}

		class_first[d] = NEW_int(nb_V);
		class_len[d] = NEW_int(nb_V);
		block_length[d] = NEW_int(nb_V);

		int_vec_classify(the_fuse[d], nb_V,
				class_first[d], class_len[d], nb_classes[d]);
		for (I = 0; I < nb_classes[d]; I++) {
			block_length[d][I] =
				tdo_scheme_get_row_class_length_fused(G, h,
				class_first[d][I], class_len[d][I]);
		}
		class_idx[d] = NEW_int(nb_classes[d]);
		if (f_v) {
			cout << "row decomposition " << d << " found "
					<< nb_classes[d] << " classes" << endl;
			cout << "class_first: ";
			Orbiter->Int_vec->print(cout, class_first[d], nb_classes[d]);
			cout << endl;
			cout << "class_len: ";
			Orbiter->Int_vec->print(cout, class_len[d], nb_classes[d]);
			cout << endl;
			cout << "block_length: ";
			Orbiter->Int_vec->print(cout, block_length[d], nb_classes[d]);
			cout << endl;
		}
		
		if (f_v) {
			cout << "computing previous scheme at depth " << d << endl;
		}
		prev_scheme[d] = NEW_int(nb_classes[d] * fuse_block_len[d]);
		for (jj = 0; jj < fuse_block_len[d]; jj++) {
			j = fuse_block_first[d] + jj;
			for (I = 0; I < nb_classes[d]; I++) {
				l = class_len[d][I];
				if (f_v) {
					cout << "j=" << j << " I=" << I
							<< " class_len[d][I]=" << l << endl;
				}
				s = 0;
				for (u = 0; u < l; u++) {
					i = class_first[d][I] + u;
					a = G.row_classes_len[h][i];
					c = scheme[i * nb_B + j];
					s += a * c;
					if (f_v) {
						cout << "i=" << i << endl;
						cout << "G.row_classes_len[h][i]="
								<< G.row_classes_len[h][i] << endl;
						cout << "scheme[i * nb_B + j]="
								<< scheme[i * nb_B + j] << endl;
						cout << "s=" << s << endl;
					}
				}
				M = G.col_classes_len[h][j];
				if (f_v) {
					cout << "G.col_classes_len[h][j]=" << M << endl;
				}
				e = s / M;
				if (e * M != s) {
					cout << "problems figuring out the "
							"previous scheme" << endl;
					cout << "d=" << d << endl;
					cout << "s=" << s << endl;
					//cout << "L=" << L << endl;
					cout << "M=" << M << endl;
					cout << "j=" << j << endl;
					cout << "I=" << I << endl;
					exit(1);
				}
				prev_scheme[d][I * fuse_block_len[d] + jj] = e;
			}
		} // next j
		if (f_v) {
			cout << "depth " << d << ", the previous "
					"col scheme is" << endl;
			Orbiter->Int_vec->print_integer_matrix_width(cout,
					prev_scheme[d], nb_classes[d],
					fuse_block_len[d], fuse_block_len[d], 4);
		}


	} // next d
	class_idx[2] = NEW_int(nb_V);
	
	//exit(1);
	
	nb_parts = 0;
	append_to_part(v + b);
	append_to_part(v);
	//part.clear();
	//part.push_back(v + b);
	//part.push_back(v);
	//nb_parts = 2;
	
	// do the column classes first:
	sum = B[0];
	for (j = 1; j < nb_B; j++) {
		class_relabel[nb_V + j] = nb_parts;
		append_to_part(v + sum);
		//part.push_back(v + sum);
		//nb_parts++;
		sum += B[j];
	}

	// do the classes for the coarse row-decomposition first:
	d = 0;
	sum = block_length[d][0];
	class_idx[d][0] = 0;
	for (I = 1; I < nb_classes[d]; I++) {
		class_idx[d][I] = nb_parts;
		ii = class_first[d][I];
		class_relabel[ii] = nb_parts;
		append_to_part(sum);
		//part.push_back(sum);
		//nb_parts++;
		sum += block_length[d][I];
	}
	prev_level[d] = nb_parts;
	if (f_v) {
		cout << "class_idx[0]=";
		Orbiter->Int_vec->print(cout, class_idx[0], nb_classes[0]);
		cout << endl;
	}
	
	// now do the classes for the fine row-decomposition:
	d = 1;
	sum = block_length[d][0];
	class_idx[d][0] = 0;
	for (I = 1; I < nb_classes[d]; I++) {
		ii = class_first[d][I];
		if (f_v) {
			cout << "I=" << I << endl;
			cout << "class_first[d][I]=ii=" << ii << endl;
			cout << "class_relabel[ii]=" << class_relabel[ii] << endl;
		}
		if (class_relabel[ii] == -1) {
			//part.push_back(sum);
			class_idx[d][I] = nb_parts;
			class_relabel[ii] = nb_parts;
			//nb_parts++;
			append_to_part(sum);
		}
		else {
			class_idx[d][I] = class_relabel[ii];
		}
		sum += block_length[d][I];
	}
	prev_level[d] = nb_parts;
	if (f_v) {
		cout << "class_idx[1]=";
		Orbiter->Int_vec->print(cout, class_idx[1], nb_classes[1]);
		cout << endl;
	}
	
	
	if (f_v) {
		cout << "prev_level[0]" << prev_level[0] << endl;
		cout << "prev_level[1]" << prev_level[1] << endl;
		cout << "the previous decomposition is " << endl;
		Orbiter->Int_vec->print(cout, part, nb_parts);
		//for (i = 0; i < nb_parts; i++) {
			//cout << part[i] << " ";
		//	}
		cout << endl;
		cout << "class_relabel: " << endl;
		Orbiter->Int_vec->print(cout, class_relabel, nb_V + nb_B);
		cout << endl;
	}
	
	nb_entries = 0;
	//entries.clear();
	d = 0;
	if (f_v) {
		cout << "writing scheme at depth " << d << " level "
				<< prev_level[d] << endl;
	}
	for (I = 0; I < nb_classes[d]; I++) {
		for (jj = 0; jj < fuse_block_len[d]; jj++) {
			j = fuse_block_first[d] + jj;
			c = prev_scheme[d][I * fuse_block_len[d] + jj];
			c1 = class_idx[d][I];
			c2 = 1 + j;
			if (f_v) {
				cout << "entry " << nb_entries << " : " << prev_level[d]
					<< " " << c2 << " " << c1 << " " << c << endl;
			}
			append_to_entries(prev_level[d], c2, c1, c);
			//nb_entries++;
		}
	}
	d = 1;
	if (f_v) {
		cout << "writing scheme at depth " << d << " level "
				<< prev_level[d] << endl;
	}
	for (I = 0; I < nb_classes[d]; I++) {
		for (jj = 0; jj < fuse_block_len[d]; jj++) {
			j = fuse_block_first[d] + jj;
			c = prev_scheme[d][I * fuse_block_len[d] + jj];
			c1 = class_idx[d][I];
			c2 = 1 + j;
			if (f_v) {
				cout << "entry " << nb_entries << " : " << prev_level[d]
					<< " " << c2 << " " << c1 << " " << c << endl;
			}
			append_to_entries(prev_level[d], c2, c1, c);
			//nb_entries++;
		}
	}
	extra_col_level = prev_level[0];
	col_level = prev_level[1];
	
	d = 2;
	
	class_idx[d][0] = 0;
	sum = 0;
	for (I = 0; I < nb_classes[1]; I++) {
		l = class_len[1][I];
		f = class_first[1][I];
		sum += G.row_classes_len[h][f + 0];
		class_idx[d][f + 0] = class_idx[d - 1][I];
		for (u = 1; u < l; u++) {
			class_relabel[f + u] = nb_parts;
			class_idx[d][f + u] = nb_parts;
			//part.push_back(sum);
			//nb_parts++;
			append_to_part(sum);
			sum += G.row_classes_len[h][f + u];
		}
	}
	if (f_v) {
		cout << "the extended decomposition is " << endl;
		Orbiter->Int_vec->print(cout, part, nb_parts);
		//for (i = 0; i < nb_parts; i++) {
			//cout << part[i] << " ";
			//}
		cout << endl;
		cout << "class_relabel: " << endl;
		Orbiter->Int_vec->print(cout, class_relabel, nb_V + nb_B);
		cout << endl;
		cout << "class_idx[2]=";
		Orbiter->Int_vec->print(cout, class_idx[2], nb_V);
		cout << endl;
	}
	row_level = nb_parts;
	for (i = 0; i < nb_V; i++) {
		for (j = 0; j < nb_B; j++) {
			c = scheme[i * nb_B + j];
			c1 = class_idx[2][i];
			c2 = 1 + j;
			if (f_v) {
				cout << "entry " << nb_entries << " : " << row_level
						<< " " << c1 << " " << c2 << " " << c << endl;
			}
			append_to_entries(row_level, c1, c2, c);
			//nb_entries++;
		}
	}

}

void geo_parameter::cut_off_two_lines(geo_parameter &GP2, 
	int *&part_relabel, int *&part_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	tdo_scheme_synthetic TDO;
	int w, j, S, i;
	int *Part;
	int *Entries;
	
	if (f_v) {
		cout << "geo_parameter::cut_off_two_lines" << endl;
	}
	if (f_vv) {
		cout << "row_level = " << row_level << endl;
		cout << "col_level = " << col_level << endl;
		cout << "extra_row_level = " << extra_row_level << endl;
		cout << "extra_col_level = " << extra_col_level << endl;
		cout << "lambda_level = " << lambda_level << endl;
	}
	Part = NEW_int(nb_parts + 1);
	Entries = NEW_int(4 * nb_entries + 1);
	for (i = 0; i < nb_parts; i++) {
		Part[i] = part[i];
	}
	Part[nb_parts] = -1;
	for (i = 0; i < 4 * nb_entries; i++) {
		Entries[i] = entries[i];
	}
	Entries[4 * nb_entries] = -1;
	
	TDO.init_TDO(Part, Entries, row_level, col_level, 
		extra_row_level, extra_col_level, 
		lambda_level, 0/*verbose_level - 1*/);

	FREE_int(Part);
	FREE_int(Entries);
	w = 0;
	for (j = TDO.nb_col_classes[COL_SCHEME] - 1; j >= 0; j--) {
		S = 0;
		for (i = 0; i < TDO.nb_row_classes[COL_SCHEME]; i++) {
			S += TDO.the_col_scheme[i * TDO.nb_col_classes[COL_SCHEME] + j];
		}
		if (S != 2) {
			break;
		}
		w += TDO.col_classes_len[COL_SCHEME][j];
	}
	if (f_v) {
		cout << "cutting off " << w << " columns" << endl;
	}
	cut_off(GP2, w, part_relabel, part_length, verbose_level);
}

#if 0
void geo_parameter::cut_off_some_lines(geo_parameter &GP2, int w, 
	int *&part_relabel, int *&part_length, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "geo_parameter::cut_off_some_lines w=" << w << endl;
		}
	cut_off(GP2, w, part_relabel, part_length, verbose_level);
}
#endif

void geo_parameter::cut_off(geo_parameter &GP2, int w,
	int *&part_relabel, int *&part_length,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);
	int i, W, a, /*u,*/ b, c, d, A, B, C;
	
	if (f_v) {
		cout << "geo_parameter::cut_off cutting off "
				"w=" << w << " columns" << endl;
	}
	if (f_vv) {
		cout << "row_level = " << row_level << endl;
		cout << "col_level = " << col_level << endl;
		cout << "extra_row_level = " << extra_row_level << endl;
		cout << "extra_col_level = " << extra_col_level << endl;
		cout << "lambda_level = " << lambda_level << endl;
		cout << "label = " << label << endl;
	}
	GP2.decomposition_type = decomposition_type;
	GP2.fuse_type = fuse_type;
	GP2.v = v;
	GP2.b = GP2.b - w;
	GP2.mode = MODE_STACK;	
	GP2.label.assign(label);
	
	part_relabel = NEW_int(nb_parts + 1);
	part_length = NEW_int(nb_parts + 1);
	W = part[0] - w;
	GP2.nb_parts = 0;
	GP2.part = NULL;
	GP2.append_to_part(W);
	//GP2.part.push_back(W);
	//W = new_part[0] = part[0] - w;
	//u = 1;
	part_relabel[0] = 0;
	part_length[0] = 0;
	for (i = 1; i <= nb_parts; i++) {
		part_length[i] = GP2.nb_parts;
		if (i == nb_parts) {
			break;
		}
		a = part[i];
		if (a >= W) {
			part_relabel[i] = -1;
		}
		else {
			part_relabel[i] = GP2.nb_parts;
			GP2.append_to_part(a);
			//GP2.part.push_back(a);
			//new_part[u] = a;
			//u++;
		}
	}
	//new_nb_parts = u;
	//GP2.nb_parts = u;
	//new_part[new_nb_parts] = -1;
	//GP2.part.push_back(-1);
	part_relabel[nb_parts] = -1;
	if (f_vv) {
		cout << "new_nb_parts = " << GP2.nb_parts << endl;
		cout << "part_relabel:" << endl;
		for (i = 0; i < nb_parts; i++) {
			cout << i << " : " << part_relabel[i] << endl;
		}
	}
	GP2.nb_entries = 0;
	GP2.entries = NULL;
	//u = 0;
	for (i = 0; i < nb_entries; i++) {
		a = entries[i * 4 + 0];
		b = entries[i * 4 + 1];
		c = entries[i * 4 + 2];
		d = entries[i * 4 + 3];
		A = part_length[a];
		B = part_relabel[b];
		C = part_relabel[c];
		if (B >= 0 && C >= 0) {
			if (f_vv) {
				cout << a << " " << b << " " << c << " " << d
						<< " -> " << A << " " << B << " " << C
						<< " " << d << endl;
			}
			GP2.append_to_entries(A, B, C, d);
			//GP2.entries.push_back(A);
			//GP2.entries.push_back(B);
			//GP2.entries.push_back(C);
			//GP2.entries.push_back(d);
			//new_entries[u * 4 + 0] = A;
			//new_entries[u * 4 + 1] = B;
			//new_entries[u * 4 + 2] = C;
			//new_entries[u * 4 + 3] = d;
			//u++;
		}
		else {
			if (f_vv) {
				cout << a << " " << b << " " << c << " " << d
						<< " eliminated" << endl;
			}
		}
	}
	//new_nb_entries = u;
	//GP2.nb_entries = u;
	//new_entries[new_nb_entries * 4] = -1;
	//GP2.entries.push_back(-1);
	if (f_vv) {
		cout << "new_nb_entries = " << GP2.nb_entries << endl;
	}
	if (row_level >= 0) {
		GP2.row_level = part_length[row_level];
	}
	else {
		GP2.row_level = row_level;
	}
	if (col_level >= 0) {
		GP2.col_level = part_length[col_level];
	}
	else {
		GP2.col_level = col_level;
	}
	GP2.extra_row_level = -1;
	GP2.extra_col_level = -1;
	if (lambda_level >= 0) {
		GP2.lambda_level = part_length[lambda_level];
	}
	else {
		GP2.lambda_level = lambda_level;
	}
#if 0
	if (extra_row_level >= 0) {
		GP2.extra_row_level = part_length[extra_row_level];
		}
	else {
		GP2.extra_row_level = extra_row_level;
		}
	if (extra_col_level >= 0) {
		GP2.extra_col_level = part_length[extra_col_level];
		}
	else {
		GP2.extra_col_level = extra_col_level;
		}
	if (lambda_level >= 0) {
		GP2.lambda_level = part_length[lambda_level];
		}
	else {
		GP2.lambda_level = lambda_level;
		}
#endif
#if 1
	if (f_v) {
		int j;
		cout << "new_part:" << endl;
		for (i = 0; i < GP2.nb_parts; i++)  {
			cout << GP2.part[i] << " ";
		}
		cout << endl;
		cout << "new_entries:" << endl;
		for (i = 0; i < GP2.nb_entries; i++) {
			for (j = 0; j < 4; j++) {
				cout << GP2.entries[i * 4 + j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
#endif
	if (f_v) {
		cout << "calling GP2.print_schemes" << endl;
		GP2.print_schemes();
	}
}

void geo_parameter::copy(geo_parameter &GP2)
{
	int i;
	
	GP2.decomposition_type = decomposition_type;
	GP2.fuse_type = fuse_type;
	GP2.v = v;
	GP2.b = b;
	GP2.mode = MODE_STACK;	
	GP2.label.assign(label);
	GP2.lambda_level = lambda_level;
	GP2.row_level = row_level;
	GP2.col_level = col_level;
	GP2.extra_row_level = extra_row_level;
	GP2.extra_col_level = extra_col_level;
	//GP2.part.clear();
	GP2.nb_parts = 0;
	for (i = 0; i < nb_parts; i++) {
		GP2.append_to_part(part[i]);
		//GP2.part.push_back(part[i]);
	}
	//GP2.part.push_back(-1);
	//GP2.nb_parts = nb_parts;
	GP2.nb_entries = 0;
	//GP2.entries.clear();
	for (i = 0; i < nb_entries; i++) {
		GP2.append_to_entries(entries[4 * i + 0],
				entries[4 * i + 1],
				entries[4 * i + 2],
				entries[4 * i + 3]);
		//GP2.entries.push_back(entries[i]);
	}
	//GP2.entries.push_back(-1);
	//GP2.nb_entries = nb_entries;
}

void geo_parameter::print_schemes()
{
	cout << "geo_parameter::print_schemes" << endl;
#if 1
	int i;
	tdo_scheme_synthetic TDO;
	int *Part, *Entries;

	Part = NEW_int(nb_parts + 1);
	Entries = NEW_int(4 * nb_entries + 1);
	for (i = 0; i < nb_parts; i++) {
		Part[i] = part[i];
	}
	Part[nb_parts] = -1;
	for (i = 0; i < 4 * nb_entries; i++) {
		Entries[i] = entries[i];
	}
	Entries[4 * nb_entries] = -1;
	
	
#if 1
	cout << "row_level=" << row_level << endl;
	cout << "col_level=" << col_level << endl;
	cout << "extra_row_level=" << extra_row_level << endl;
	cout << "extra_col_level=" << extra_col_level << endl;
	cout << "lambda_level=" << lambda_level << endl;
	cout << "geo_parameter::print_schemes "
			"before TDO.init_TDO" << endl;
#endif
	TDO.init_TDO(Part, Entries, row_level, col_level, 
		extra_row_level, extra_col_level, 
		lambda_level, 0/*verbose_level - 1*/);
	//cout << "geo_parameter::print_schemes
	//after TDO.init_TDO" << endl;
			
	FREE_int(Part);
	FREE_int(Entries);

	TDO.print_all_schemes();
	cout << "geo_parameter::print_schemes after "
			"TDO.print_all_schemes" << endl;
#endif
}

}}}



