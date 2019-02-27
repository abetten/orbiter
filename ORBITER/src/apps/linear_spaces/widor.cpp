// widor.C
// Anton Betten
//
// converts a "Widor" file to a tdo file
//
// started:  May 6, 2008

#include "orbiter.h"

using namespace std;


using namespace orbiter;



int main(int argc, char **argv)
{
	int i;
	char *widor_fname;
	char str[1000];
	char ext[1000];
	char fname_out[1000];
	char label[1000];
	int verbose_level = 0;
		
	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	widor_fname = argv[argc - 1];
	strcpy(str, widor_fname);
	get_extension_if_present(str, ext);
	chop_off_extension_if_present(str, ext);
	sprintf(fname_out, "%s.tdo", str);

	cout << "reading file " << widor_fname << endl;
	{
	geo_parameter GP;
	tdo_scheme G;
	ifstream f(widor_fname);
	ofstream g(fname_out);
	for (i = 0; ; i++) {
		if (f.eof()) {
			cout << "end of file reached" << endl;
			break;
			}
		if (!GP.input(f)) {
			cout << "GP.input returns false" << endl;
			break;
			}
		if (f_v) {
			cout << "read decomposition " << i 
				<< " v=" << GP.v << " b=" << GP.b << endl;
			}
		GP.convert_single_to_stack(verbose_level - 1);
		if (f_v) {
			cout << "after convert_single_to_stack" << endl;
			}
		if (strlen(GP.label)) {
			sprintf(label, "%s", GP.label);
			}
		else {
			sprintf(label, "%d", i);
			}
		GP.write(g, label);
		if (f_v) {
			cout << "after write" << endl;
			}
		GP.init_tdo_scheme(G, verbose_level - 1);
		if (f_v) {
			cout << "after init_tdo_scheme" << endl;
			}
		if (f_vv) {
			GP.print_schemes(G);
			}
		}
	g << "-1 " << i << endl;
	}
	cout << "written file " << fname_out << " of size " << file_size(fname_out) << endl;
}

