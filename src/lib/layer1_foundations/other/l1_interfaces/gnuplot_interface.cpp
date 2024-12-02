/*
 * gnuplot_interface.cpp
 *
 *  Created on: Jun 27, 2023
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace l1_interfaces {


gnuplot_interface::gnuplot_interface()
{
	Record_birth();

}

gnuplot_interface::~gnuplot_interface()
{
	Record_death();

}



void gnuplot_interface::gnuplot(
		std::string &data_file_csv,
		std::string &title,
		std::string &label_x,
		std::string &label_y,
		int verbose_level)
{

	orbiter_kernel_system::file_io Fio;


	cout << "gnuplot_interface::gnuplot "
			"Reading data from file " << data_file_csv << " of size "
			<< Fio.file_size(data_file_csv) << endl;


	data_structures::spreadsheet *S;

	S = NEW_OBJECT(data_structures::spreadsheet);

	S->read_spreadsheet(data_file_csv, 0 /*verbose_level*/);

	int m, n;

	m = S->nb_rows;
	n = S->nb_cols;


#if 0
	int *Data;
	int m, n;

	Fio.int_matrix_read_csv(data_file_csv, Data, m, n, verbose_level);

	Int_matrix_print(Data, m, n);


	cout << "Data:" << endl;
	Int_matrix_print(Data, m, n);
#endif



	string prefix;

	prefix = data_file_csv;

	data_structures::string_tools ST;

	ST.chop_off_extension(prefix);

	cout << "gnuplot_interface::gnuplot "
			"prefix=" << prefix << endl;




	string fname1;
	string fname2;
	string fname3;
	string cmd;

	fname1 = prefix + "_data.dat";
	fname2 = prefix + "_gnuplot.txt";
	fname3 = prefix + "_gnuplot.png";
	{
		ofstream ost(fname1);

		ost << "# " << fname1 << endl;

		int i, j;
		for (j = 0; j < n; j++) {

			string s;


			S->get_string(s, 0, j);

			ost << s << "\t";
		}
		ost << endl;


		for (i = 1; i < m; i++) {

			string s;


			for (j = 0; j < n; j++) {
				S->get_string(s, i, j);

				ost << s << "\t";
			}
			ost << endl;
		}

	}
	cout << "gnuplot_interface::gnuplot "
			"Written file " << fname1 << " of size " << Fio.file_size(fname1) << endl;

	{
		ofstream ost(fname2);

		ost << "# " << fname2 << endl;
		ost << "set terminal png linewidth 3" << endl;
		ost << "set title '" << title << "'" << endl;
		ost << "set style data linespoints" << endl;
		ost << "set style line 2 lc rgb '#dd181f' pt 2 lt 2 lw 2 # red" << endl;
		ost << "set style line 3 lc rgb '#0060ad' pt 1 lt 1 lw 2 # blue" << endl;
		ost << "set style line 4 lc rgb '#ff9933' pt 3 lt 3 lw 2 # orange" << endl;
		ost << "set style line 5 lc rgb '#ffcc33' pt 4 lt 4 lw 2 # light orange" << endl;
		ost << "set style line 6 lc rgb '#336600' pt 5 lt 5 lw 2 # green" << endl;
		ost << "set style line 7 lc rgb '#9900CC' pt 6 lt 6 lw 2 # purple" << endl;
		ost << "set style line 8 lc rgb '#000000' pt 7 lt 7 lw 2 # black" << endl;
		//ost << "set style dots line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5   # --- red" << endl;
		//ost << "set logscale xy" << endl;
		ost << "set xlabel '" << label_x << "'" << endl;
		ost << "set ylabel '" << label_y << "'" << endl;
		ost << "set key top right" << endl;
		ost << "plot '" << fname1 << "' using 2:xtic(1) title columnheader(2) ls 2, \\" << endl;
		ost << "for [i=3:" << n << "] '' using i title columnheader(i) ls i "<< endl;
	}
	cout << "Written file " << fname2 << " of size " << Fio.file_size(fname2) << endl;

	cmd = "gnuplot " + fname2 + " >" + fname3;
	cout << "Executing command: " << cmd << endl;
	system(cmd.c_str());

	FREE_OBJECT(S);



#if 0
//https://gnuplot.sourceforge.net/demo_5.4/histograms.html
#
# Example of using histogram modes
#
set title "US immigration from Europe by decade"
set datafile missing "-"
set xtics nomirror rotate by -45
set key noenhanced
#
# First plot using linespoints
set style data linespoints
plot 'immigration.dat' using 2:xtic(1) title columnheader(2), \
for [i=3:22] '' using i title columnheader(i)
#endif


#if 0
SO72347937_Styles.gp:
set style line 1 lc rgb '#0060ad' pt 1 lt 1 lw 2 # blue
set style line 2 lc rgb '#dd181f' pt 2 lt 2 lw 2 # red
set style line 3 lc rgb '#ff9933' pt 3 lt 3 lw 2 # orange
set style line 4 lc rgb '#ffcc33' pt 4 lt 4 lw 2 # light orange
set style line 5 lc rgb '#336600' pt 5 lt 5 lw 2 # green
set style line 6 lc rgb '#9900CC' pt 6 lt 6 lw 2 # purple
set style line 7 lc rgb '#000000' pt 7 lt 7 lw 2 # black

### load custom style file
reset session

load "SO72347937_Styles.gp"

set key top left
set samples 20

plot for [i=1:7] '+' u 1:($1*i) w lp ls i ti sprintf("Style %d",i)
### end of code
#endif


#if 0

# IMMIGRATION BY REGION AND SELECTED COUNTRY OF LAST RESIDENCE
#
Region	Austria	Hungary	Belgium	Czechoslovakia	Denmark	France	Germany	Greece	Ireland	Italy	Netherlands	Norway	Sweden	Poland	Portugal	Romania	Soviet_Union	Spain	Switzerland	United_Kingdom	Yugoslavia	Other_Europe	TOTAL
1891-1900	234081	181288	18167	-	50231	30770	505152	15979	388416	651893	26758	95015	226266	96720	27508	12750	505290	8731	31179	271538	-	282	3378014
1901-1910	668209	808511	41635	-	65285	73379	341498	167519	339065	2045877	48262	190505	249534	-	69149	53008	1597306	27935	34922	525950	-	39945	7387494
1911-1920	453649	442693	33746	3426	41983	61897	143945	184201	146181	1109524	43718	66395	95074	4813	89732	13311	921201	68611	23091	341408	1888	31400	4321887
1921-1930	32868	30680	15846	102194	32430	49610	412202	51084	211234	455315	26948	68531	97249	227734	29994	67646	61742	28958	29676	339570	49064	42619	2463194
1931-1940	3563	7861	4817	14393	2559	12623	144058	9119	10973	68028	7150	4740	3960	17026	3329	3871	1370	3258	5512	31572	5835	11949	377566
1941-1950	24860	3469	12189	8347	5393	38809	226578	8973	19789	57661	14860	10100	10665	7571	7423	1076	571	2898	10547	139306	1576	8486	621147
1951-1960	67106	36637	18575	918	10984	51121	477765	47608	43362	185491	52277	22935	21697	9985	19588	1039	671	7894	17675	202824	8225	16350	1325727
1961-1970	20621	5401	9192	3273	9201	45237	190796	85969	32966	214111	30606	15484	17116	53539	76065	3531	2465	44659	18453	213822	20381	11604	1124492


#endif


#if 0
//old version of gnuplot (around 2018):
# plotting_data3.dat
# First data block (index 0)
# X   Y
  1   2
  2   3


# Second index block (index 1)
# X   Y
  3   2
  4   1
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5   # --- blue
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5   # --- red
plot 'plotting-data3.dat' index 0 with linespoints ls 1, \
     ''                   index 1 with linespoints ls 2
#endif




}



}}}}




