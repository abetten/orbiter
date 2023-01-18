/*
 * latex_interface.cpp
 *
 *  Created on: Apr 13, 2019
 *      Author: betten
 */


#include "foundations.h"

#include <sstream>

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace orbiter_kernel_system {


latex_interface::latex_interface()
{

}

latex_interface::~latex_interface()
{

}

void latex_interface::head_easy(std::ostream& ost)
{
	std::string dummy;

	dummy.assign("");

	head(ost,
		FALSE /* f_book */,
		FALSE /* f_title */,
		dummy, dummy,
		FALSE /*f_toc */,
		FALSE /* f_landscape */,
		FALSE /* f_12pt */,
		FALSE /* f_enlarged_page */,
		TRUE /* f_pagenumbers */,
		dummy /* extras_for_preamble */);

}

void latex_interface::head_easy_with_extras_in_the_praeamble(
		std::ostream& ost, std::string &extras)
{
	std::string dummy;

	dummy.assign("");

	head(ost,
		FALSE /* f_book */,
		FALSE /* f_title */,
		dummy, dummy,
		FALSE /*f_toc */,
		FALSE /* f_landscape */,
		FALSE /* f_12pt */,
		FALSE /* f_enlarged_page */,
		TRUE /* f_pagenumbers */,
		extras /* extras_for_preamble */);

}

void latex_interface::head_easy_sideways(std::ostream& ost)
{
	std::string dummy;

	dummy.assign("");

	head(ost, FALSE /* f_book */,
		FALSE /* f_title */,
		dummy, dummy,
		FALSE /*f_toc */,
		TRUE /* f_landscape */,
		FALSE /* f_12pt */,
		FALSE /* f_enlarged_page */,
		TRUE /* f_pagenumbers */,
		dummy /* extras_for_preamble */);

}

void latex_interface::head(
		std::ostream& ost,
	int f_book, int f_title,
	std::string &title, std::string &author,
	int f_toc, int f_landscape, int f_12pt,
	int f_enlarged_page, int f_pagenumbers,
	std::string &extras_for_preamble)
{
	if (f_12pt) {
		ost << "\\documentclass[12pt]{";
		}
	else {
		ost << "\\documentclass{";
		}
	if (f_book)
		ost << "book";
	else
		ost << "article";
	ost << "}" << endl;
	ost << "% a4paper" << endl;
	ost << endl;
	ost << "%\\usepackage[dvips]{epsfig}" << endl;
	ost << "%\\usepackage{cours11, cours}" << endl;
	ost << "%\\usepackage{fancyheadings}" << endl;
	ost << "%\\usepackage{calc}" << endl;
	ost << "\\usepackage{amsmath}" << endl;
	ost << "\\usepackage{amssymb}" << endl;
	ost << "\\usepackage{latexsym}" << endl;
	ost << "\\usepackage{epsfig}" << endl;
	ost << "\\usepackage{enumerate}" << endl;
	ost << "%\\usepackage{supertabular}" << endl;
	ost << "%\\usepackage{wrapfig}" << endl;
	ost << "%\\usepackage{blackbrd}" << endl;
	ost << "%\\usepackage{epic,eepic}" << endl;
	ost << "\\usepackage{rotating}" << endl;
	ost << "\\usepackage{multicol}" << endl;
	ost << "%\\usepackage{multirow}" << endl;
	ost << "\\usepackage{makeidx} % additional command see" << endl;
	ost << "\\usepackage{rotating}" << endl;
	ost << "\\usepackage{array}" << endl;
	ost << "\\usepackage{tikz}" << endl;
	ost << "\\usepackage{longtable}" << endl;
	ost << "\\usepackage{anyfontsize}" << endl;
	ost << "\\usepackage{t1enc}" << endl;
	ost << "%\\usepackage{amsmath,amsfonts}" << endl;
	ost << endl;
	ost << endl;
	ost << "%\\usepackage[mtbold,mtplusscr]{mathtime}" << endl;
	ost << "% lucidacal,lucidascr," << endl;
	ost << endl;
	ost << "%\\usepackage{mathtimy}" << endl;
	ost << "%\\usepackage{bm}" << endl;
	ost << "%\\usepackage{avant}" << endl;
	ost << "%\\usepackage{basker}" << endl;
	ost << "%\\usepackage{bembo}" << endl;
	ost << "%\\usepackage{bookman}" << endl;
	ost << "%\\usepackage{chancery}" << endl;
	ost << "%\\usepackage{garamond}" << endl;
	ost << "%\\usepackage{helvet}" << endl;
	ost << "%\\usepackage{newcent}" << endl;
	ost << "%\\usepackage{palatino}" << endl;
	ost << "%\\usepackage{times}" << endl;
	ost << "%\\usepackage{pifont}" << endl;
	if (f_enlarged_page) {
		ost << "\\usepackage{fullpage}" << endl;
		ost << "\\usepackage[top=1in,bottom=0.2in,right=1in,left=1in]{geometry}" << endl; // A Betten 2/7/2021
#if 0
		ost << "%\\voffset=-1.5cm" << endl;
		ost << "\\hoffset=-2cm" << endl;
		ost << "\\textwidth=20cm" << endl;
		ost << "%\\topmargin 0.0in" << endl;
		ost << "\\textheight 25cm" << endl;
#endif
		}

	if (extras_for_preamble.length()) {
		ost << extras_for_preamble << endl;
		}
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\parindent=0pt\n";
	ost << endl;
	//ost << "\\renewcommand{\\baselinestretch}{1.5}\n";
	ost << endl;


#if 0
	if (f_enlarged_page) {
		ost << "\\hoffset -2cm\n";
		ost << "\\voffset -1cm\n";
		ost << "\\topmargin 0.0cm\n";
		if (f_landscape) {
			ost << "\\textheight=18cm\n";
			ost << "\\textwidth=23cm\n";
			}
		else {
			ost << "\\textheight=23cm\n";
			ost << "\\textwidth=18cm\n";
			}
		}
	else {
		ost << "\\hoffset -0.7cm\n";
		ost << "%\\voffset 0cm\n";
		ost << endl;
		ost << "%\\oddsidemargin=15pt\n";
		ost << endl;
		ost << "%\\oddsidemargin 0pt\n";
		ost << "%\\evensidemargin 0pt\n";
		ost << "%\\topmargin 0pt\n";
		ost << endl;
#if 1
		if (f_landscape) {
			ost << "\\textwidth = 20cm\n";
			ost << "\\textheight= 17cm\n";
			}
		else {
			ost << "\\textwidth = 17cm\n";
			ost << "\\textheight= 21cm\n";
			}
		ost << endl;
#endif
		}
#endif


	ost << "%\\topmargin=0pt\n";
	ost << "%\\headsep=18pt\n";
	ost << "%\\footskip=45pt\n";
	ost << "%\\mathsurround=1pt\n";
	ost << "%\\evensidemargin=0pt\n";
	ost << "%\\oddsidemargin=15pt\n";
	ost << endl;

	ost << "%\\setlength{\\textheight}{\\baselineskip*41+\\topskip}\n";
	ost << endl;


	ost << "\\newcommand{\\sectionline}{" << endl;
	ost << "   \\nointerlineskip \\vspace{\\baselineskip}" << endl;
	ost << "   \\hspace{\\fill}\\rule{0.9\\linewidth}{1.7pt}\\hspace{\\fill}" << endl;
	ost << "   \\par\\nointerlineskip \\vspace{\\baselineskip}" << endl;
	ost << "   }" << endl;

	ost << "\\newcommand\\setTBstruts{\\def\\T{\\rule{0pt}{2.6ex}}%" << endl;
	ost << "\\def\\B{\\rule[-1.2ex]{0pt}{0pt}}}" << endl;

	ost << "\\newcommand{\\ans}[1]{\\\\{\\bf ANSWER}: {#1}}" << endl;
	ost << "\\newcommand{\\Aut}{{\\rm Aut}}\n";
	ost << "\\newcommand{\\Sym}{{\\rm Sym}}\n";
	ost << "\\newcommand{\\sFix}{{\\cal Fix}}\n";
	ost << "\\newcommand{\\sOrbits}{{\\cal Orbits}}\n";
	//ost << "\\newcommand{\\sFix}{{\\mathscr Fix}}\n";
	//ost << "\\newcommand{\\sOrbits}{{\\mathscr Orbits}}\n";
	ost << "\\newcommand{\\Stab}{{\\rm Stab}}\n";
	ost << "\\newcommand{\\Fix}{{\\rm Fix}}\n";
	ost << "\\newcommand{\\fix}{{\\rm fix}}\n";
	ost << "\\newcommand{\\Orbits}{{\\rm Orbits}}\n";
	ost << "\\newcommand{\\PG}{{\\rm PG}}\n";
	ost << "\\newcommand{\\AG}{{\\rm AG}}\n";
	ost << "\\newcommand{\\SQS}{{\\rm SQS}}\n";
	ost << "\\newcommand{\\STS}{{\\rm STS}}\n";
	//ost << "\\newcommand{\\Sp}{{\\rm Sp}}\n";
	ost << "\\newcommand{\\PSL}{{\\rm PSL}}\n";
	ost << "\\newcommand{\\PGL}{{\\rm PGL}}\n";
	ost << "\\newcommand{\\PSSL}{{\\rm P\\Sigma L}}\n";
	ost << "\\newcommand{\\PGGL}{{\\rm P\\Gamma L}}\n";
	ost << "\\newcommand{\\SL}{{\\rm SL}}\n";
	ost << "\\newcommand{\\GL}{{\\rm GL}}\n";
	ost << "\\newcommand{\\SSL}{{\\rm \\Sigma L}}\n";
	ost << "\\newcommand{\\GGL}{{\\rm \\Gamma L}}\n";
	ost << "\\newcommand{\\ASL}{{\\rm ASL}}\n";
	ost << "\\newcommand{\\AGL}{{\\rm AGL}}\n";
	ost << "\\newcommand{\\ASSL}{{\\rm A\\Sigma L}}\n";
	ost << "\\newcommand{\\AGGL}{{\\rm A\\Gamma L}}\n";
	ost << "\\newcommand{\\PSU}{{\\rm PSU}}\n";
	ost << "\\newcommand{\\HS}{{\\rm HS}}\n";
	ost << "\\newcommand{\\Hol}{{\\rm Hol}}\n";
	ost << "\\newcommand{\\SO}{{\\rm SO}}\n";
	ost << "\\newcommand{\\ASO}{{\\rm ASO}}\n";

	ost << "\\newcommand{\\la}{\\langle}\n";
	ost << "\\newcommand{\\ra}{\\rangle}\n";


	ost << "\\newcommand{\\cA}{{\\cal A}}\n";
	ost << "\\newcommand{\\cB}{{\\cal B}}\n";
	ost << "\\newcommand{\\cC}{{\\cal C}}\n";
	ost << "\\newcommand{\\cD}{{\\cal D}}\n";
	ost << "\\newcommand{\\cE}{{\\cal E}}\n";
	ost << "\\newcommand{\\cF}{{\\cal F}}\n";
	ost << "\\newcommand{\\cG}{{\\cal G}}\n";
	ost << "\\newcommand{\\cH}{{\\cal H}}\n";
	ost << "\\newcommand{\\cI}{{\\cal I}}\n";
	ost << "\\newcommand{\\cJ}{{\\cal J}}\n";
	ost << "\\newcommand{\\cK}{{\\cal K}}\n";
	ost << "\\newcommand{\\cL}{{\\cal L}}\n";
	ost << "\\newcommand{\\cM}{{\\cal M}}\n";
	ost << "\\newcommand{\\cN}{{\\cal N}}\n";
	ost << "\\newcommand{\\cO}{{\\cal O}}\n";
	ost << "\\newcommand{\\cP}{{\\cal P}}\n";
	ost << "\\newcommand{\\cQ}{{\\cal Q}}\n";
	ost << "\\newcommand{\\cR}{{\\cal R}}\n";
	ost << "\\newcommand{\\cS}{{\\cal S}}\n";
	ost << "\\newcommand{\\cT}{{\\cal T}}\n";
	ost << "\\newcommand{\\cU}{{\\cal U}}\n";
	ost << "\\newcommand{\\cV}{{\\cal V}}\n";
	ost << "\\newcommand{\\cW}{{\\cal W}}\n";
	ost << "\\newcommand{\\cX}{{\\cal X}}\n";
	ost << "\\newcommand{\\cY}{{\\cal Y}}\n";
	ost << "\\newcommand{\\cZ}{{\\cal Z}}\n";

	ost << "\\newcommand{\\rmA}{{\\rm A}}\n";
	ost << "\\newcommand{\\rmB}{{\\rm B}}\n";
	ost << "\\newcommand{\\rmC}{{\\rm C}}\n";
	ost << "\\newcommand{\\rmD}{{\\rm D}}\n";
	ost << "\\newcommand{\\rmE}{{\\rm E}}\n";
	ost << "\\newcommand{\\rmF}{{\\rm F}}\n";
	ost << "\\newcommand{\\rmG}{{\\rm G}}\n";
	ost << "\\newcommand{\\rmH}{{\\rm H}}\n";
	ost << "\\newcommand{\\rmI}{{\\rm I}}\n";
	ost << "\\newcommand{\\rmJ}{{\\rm J}}\n";
	ost << "\\newcommand{\\rmK}{{\\rm K}}\n";
	ost << "\\newcommand{\\rmL}{{\\rm L}}\n";
	ost << "\\newcommand{\\rmM}{{\\rm M}}\n";
	ost << "\\newcommand{\\rmN}{{\\rm N}}\n";
	ost << "\\newcommand{\\rmO}{{\\rm O}}\n";
	ost << "\\newcommand{\\rmP}{{\\rm P}}\n";
	ost << "\\newcommand{\\rmQ}{{\\rm Q}}\n";
	ost << "\\newcommand{\\rmR}{{\\rm R}}\n";
	ost << "\\newcommand{\\rmS}{{\\rm S}}\n";
	ost << "\\newcommand{\\rmT}{{\\rm T}}\n";
	ost << "\\newcommand{\\rmU}{{\\rm U}}\n";
	ost << "\\newcommand{\\rmV}{{\\rm V}}\n";
	ost << "\\newcommand{\\rmW}{{\\rm W}}\n";
	ost << "\\newcommand{\\rmX}{{\\rm X}}\n";
	ost << "\\newcommand{\\rmY}{{\\rm Y}}\n";
	ost << "\\newcommand{\\rmZ}{{\\rm Z}}\n";

	ost << "\\newcommand{\\bA}{{\\bf A}}\n";
	ost << "\\newcommand{\\bB}{{\\bf B}}\n";
	ost << "\\newcommand{\\bC}{{\\bf C}}\n";
	ost << "\\newcommand{\\bD}{{\\bf D}}\n";
	ost << "\\newcommand{\\bE}{{\\bf E}}\n";
	ost << "\\newcommand{\\bF}{{\\bf F}}\n";
	ost << "\\newcommand{\\bG}{{\\bf G}}\n";
	ost << "\\newcommand{\\bH}{{\\bf H}}\n";
	ost << "\\newcommand{\\bI}{{\\bf I}}\n";
	ost << "\\newcommand{\\bJ}{{\\bf J}}\n";
	ost << "\\newcommand{\\bK}{{\\bf K}}\n";
	ost << "\\newcommand{\\bL}{{\\bf L}}\n";
	ost << "\\newcommand{\\bM}{{\\bf M}}\n";
	ost << "\\newcommand{\\bN}{{\\bf N}}\n";
	ost << "\\newcommand{\\bO}{{\\bf O}}\n";
	ost << "\\newcommand{\\bP}{{\\bf P}}\n";
	ost << "\\newcommand{\\bQ}{{\\bf Q}}\n";
	ost << "\\newcommand{\\bR}{{\\bf R}}\n";
	ost << "\\newcommand{\\bS}{{\\bf S}}\n";
	ost << "\\newcommand{\\bT}{{\\bf T}}\n";
	ost << "\\newcommand{\\bU}{{\\bf U}}\n";
	ost << "\\newcommand{\\bV}{{\\bf V}}\n";
	ost << "\\newcommand{\\bW}{{\\bf W}}\n";
	ost << "\\newcommand{\\bX}{{\\bf X}}\n";
	ost << "\\newcommand{\\bY}{{\\bf Y}}\n";
	ost << "\\newcommand{\\bZ}{{\\bf Z}}\n";

#if 0
	ost << "\\newcommand{\\sA}{{\\mathscr A}}\n";
	ost << "\\newcommand{\\sB}{{\\mathscr B}}\n";
	ost << "\\newcommand{\\sC}{{\\mathscr C}}\n";
	ost << "\\newcommand{\\sD}{{\\mathscr D}}\n";
	ost << "\\newcommand{\\sE}{{\\mathscr E}}\n";
	ost << "\\newcommand{\\sF}{{\\mathscr F}}\n";
	ost << "\\newcommand{\\sG}{{\\mathscr G}}\n";
	ost << "\\newcommand{\\sH}{{\\mathscr H}}\n";
	ost << "\\newcommand{\\sI}{{\\mathscr I}}\n";
	ost << "\\newcommand{\\sJ}{{\\mathscr J}}\n";
	ost << "\\newcommand{\\sK}{{\\mathscr K}}\n";
	ost << "\\newcommand{\\sL}{{\\mathscr L}}\n";
	ost << "\\newcommand{\\sM}{{\\mathscr M}}\n";
	ost << "\\newcommand{\\sN}{{\\mathscr N}}\n";
	ost << "\\newcommand{\\sO}{{\\mathscr O}}\n";
	ost << "\\newcommand{\\sP}{{\\mathscr P}}\n";
	ost << "\\newcommand{\\sQ}{{\\mathscr Q}}\n";
	ost << "\\newcommand{\\sR}{{\\mathscr R}}\n";
	ost << "\\newcommand{\\sS}{{\\mathscr S}}\n";
	ost << "\\newcommand{\\sT}{{\\mathscr T}}\n";
	ost << "\\newcommand{\\sU}{{\\mathscr U}}\n";
	ost << "\\newcommand{\\sV}{{\\mathscr V}}\n";
	ost << "\\newcommand{\\sW}{{\\mathscr W}}\n";
	ost << "\\newcommand{\\sX}{{\\mathscr X}}\n";
	ost << "\\newcommand{\\sY}{{\\mathscr Y}}\n";
	ost << "\\newcommand{\\sZ}{{\\mathscr Z}}\n";
#else
	ost << "\\newcommand{\\sA}{{\\cal A}}\n";
	ost << "\\newcommand{\\sB}{{\\cal B}}\n";
	ost << "\\newcommand{\\sC}{{\\cal C}}\n";
	ost << "\\newcommand{\\sD}{{\\cal D}}\n";
	ost << "\\newcommand{\\sE}{{\\cal E}}\n";
	ost << "\\newcommand{\\sF}{{\\cal F}}\n";
	ost << "\\newcommand{\\sG}{{\\cal G}}\n";
	ost << "\\newcommand{\\sH}{{\\cal H}}\n";
	ost << "\\newcommand{\\sI}{{\\cal I}}\n";
	ost << "\\newcommand{\\sJ}{{\\cal J}}\n";
	ost << "\\newcommand{\\sK}{{\\cal K}}\n";
	ost << "\\newcommand{\\sL}{{\\cal L}}\n";
	ost << "\\newcommand{\\sM}{{\\cal M}}\n";
	ost << "\\newcommand{\\sN}{{\\cal N}}\n";
	ost << "\\newcommand{\\sO}{{\\cal O}}\n";
	ost << "\\newcommand{\\sP}{{\\cal P}}\n";
	ost << "\\newcommand{\\sQ}{{\\cal Q}}\n";
	ost << "\\newcommand{\\sR}{{\\cal R}}\n";
	ost << "\\newcommand{\\sS}{{\\cal S}}\n";
	ost << "\\newcommand{\\sT}{{\\cal T}}\n";
	ost << "\\newcommand{\\sU}{{\\cal U}}\n";
	ost << "\\newcommand{\\sV}{{\\cal V}}\n";
	ost << "\\newcommand{\\sW}{{\\cal W}}\n";
	ost << "\\newcommand{\\sX}{{\\cal X}}\n";
	ost << "\\newcommand{\\sY}{{\\cal Y}}\n";
	ost << "\\newcommand{\\sZ}{{\\cal Z}}\n";
#endif

	ost << "\\newcommand{\\frakA}{{\\mathfrak A}}\n";
	ost << "\\newcommand{\\frakB}{{\\mathfrak B}}\n";
	ost << "\\newcommand{\\frakC}{{\\mathfrak C}}\n";
	ost << "\\newcommand{\\frakD}{{\\mathfrak D}}\n";
	ost << "\\newcommand{\\frakE}{{\\mathfrak E}}\n";
	ost << "\\newcommand{\\frakF}{{\\mathfrak F}}\n";
	ost << "\\newcommand{\\frakG}{{\\mathfrak G}}\n";
	ost << "\\newcommand{\\frakH}{{\\mathfrak H}}\n";
	ost << "\\newcommand{\\frakI}{{\\mathfrak I}}\n";
	ost << "\\newcommand{\\frakJ}{{\\mathfrak J}}\n";
	ost << "\\newcommand{\\frakK}{{\\mathfrak K}}\n";
	ost << "\\newcommand{\\frakL}{{\\mathfrak L}}\n";
	ost << "\\newcommand{\\frakM}{{\\mathfrak M}}\n";
	ost << "\\newcommand{\\frakN}{{\\mathfrak N}}\n";
	ost << "\\newcommand{\\frakO}{{\\mathfrak O}}\n";
	ost << "\\newcommand{\\frakP}{{\\mathfrak P}}\n";
	ost << "\\newcommand{\\frakQ}{{\\mathfrak Q}}\n";
	ost << "\\newcommand{\\frakR}{{\\mathfrak R}}\n";
	ost << "\\newcommand{\\frakS}{{\\mathfrak S}}\n";
	ost << "\\newcommand{\\frakT}{{\\mathfrak T}}\n";
	ost << "\\newcommand{\\frakU}{{\\mathfrak U}}\n";
	ost << "\\newcommand{\\frakV}{{\\mathfrak V}}\n";
	ost << "\\newcommand{\\frakW}{{\\mathfrak W}}\n";
	ost << "\\newcommand{\\frakX}{{\\mathfrak X}}\n";
	ost << "\\newcommand{\\frakY}{{\\mathfrak Y}}\n";
	ost << "\\newcommand{\\frakZ}{{\\mathfrak Z}}\n";

	ost << "\\newcommand{\\fraka}{{\\mathfrak a}}\n";
	ost << "\\newcommand{\\frakb}{{\\mathfrak b}}\n";
	ost << "\\newcommand{\\frakc}{{\\mathfrak c}}\n";
	ost << "\\newcommand{\\frakd}{{\\mathfrak d}}\n";
	ost << "\\newcommand{\\frake}{{\\mathfrak e}}\n";
	ost << "\\newcommand{\\frakf}{{\\mathfrak f}}\n";
	ost << "\\newcommand{\\frakg}{{\\mathfrak g}}\n";
	ost << "\\newcommand{\\frakh}{{\\mathfrak h}}\n";
	ost << "\\newcommand{\\fraki}{{\\mathfrak i}}\n";
	ost << "\\newcommand{\\frakj}{{\\mathfrak j}}\n";
	ost << "\\newcommand{\\frakk}{{\\mathfrak k}}\n";
	ost << "\\newcommand{\\frakl}{{\\mathfrak l}}\n";
	ost << "\\newcommand{\\frakm}{{\\mathfrak m}}\n";
	ost << "\\newcommand{\\frakn}{{\\mathfrak n}}\n";
	ost << "\\newcommand{\\frako}{{\\mathfrak o}}\n";
	ost << "\\newcommand{\\frakp}{{\\mathfrak p}}\n";
	ost << "\\newcommand{\\frakq}{{\\mathfrak q}}\n";
	ost << "\\newcommand{\\frakr}{{\\mathfrak r}}\n";
	ost << "\\newcommand{\\fraks}{{\\mathfrak s}}\n";
	ost << "\\newcommand{\\frakt}{{\\mathfrak t}}\n";
	ost << "\\newcommand{\\fraku}{{\\mathfrak u}}\n";
	ost << "\\newcommand{\\frakv}{{\\mathfrak v}}\n";
	ost << "\\newcommand{\\frakw}{{\\mathfrak w}}\n";
	ost << "\\newcommand{\\frakx}{{\\mathfrak x}}\n";
	ost << "\\newcommand{\\fraky}{{\\mathfrak y}}\n";
	ost << "\\newcommand{\\frakz}{{\\mathfrak z}}\n";


	ost << "\\newcommand{\\Tetra}{{\\mathfrak Tetra}}\n";
	ost << "\\newcommand{\\Cube}{{\\mathfrak Cube}}\n";
	ost << "\\newcommand{\\Octa}{{\\mathfrak Octa}}\n";
	ost << "\\newcommand{\\Dode}{{\\mathfrak Dode}}\n";
	ost << "\\newcommand{\\Ico}{{\\mathfrak Ico}}\n";

	ost << "\\newcommand{\\bbF}{{\\mathbb F}}\n";
	ost << "\\newcommand{\\bbQ}{{\\mathbb Q}}\n";
	ost << "\\newcommand{\\bbC}{{\\mathbb C}}\n";
	ost << "\\newcommand{\\bbR}{{\\mathbb R}}\n";

	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\makeindex\n";
	ost << endl;
	ost << "\\begin{document} \n";
	ost << "\\setTBstruts" << endl;
	ost << endl;
	ost << "\\bibliographystyle{plain}\n";
	if (!f_pagenumbers) {
		ost << "\\pagestyle{empty}\n";
		}
	ost << "%\\large\n";
	ost << endl;
	ost << "{\\allowdisplaybreaks%\n";
	ost << endl;
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\makeindex\n";
	ost << endl;
	ost << "%\\renewcommand{\\labelenumi}{(\\roman{enumi})}\n";
	ost << endl;

	if (f_title) {
		ost << "\\title{" << title << "}\n";
		ost << "\\author{" << author << "}%end author\n";
		ost << "%\\date{}\n";
		ost << "\\maketitle%\n";
		}
	ost << "%\\pagenumbering{roman}\n";
	ost << "%\\thispagestyle{empty}\n";
	if (f_toc) {
		ost << "\\tableofcontents\n";
		}
	ost << "%\\input et.tex%\n";
	ost << "%\\thispagestyle{empty}%\\phantom{page2}%\\clearpage%\n";
	ost << "%\\addcontentsline{toc}{chapter}{Inhaltsverzeichnis}%\n";
	ost << "%\\tableofcontents\n";
	ost << "%\\listofsymbols\n";
	if (f_toc){
		ost << "\\clearpage\n";
		ost << endl;
		}
	ost << "%\\pagenumbering{arabic}\n";
	ost << "%\\pagenumbering{roman}\n";
	ost << endl;
	ost << endl;
	ost << endl;
}


void latex_interface::foot(std::ostream& ost)
{
	ost << endl;
	ost << endl;
	ost << "%\\bibliographystyle{gerplain}% wird oben eingestellt\n";
	ost << "%\\addcontentsline{toc}{section}{References}\n";
	ost << "%\\bibliography{../MY_BIBLIOGRAPHY/anton}\n";
	ost << "% ACHTUNG: nicht vergessen:\n";
	ost << "% die Zeile\n";
	ost << "%\\addcontentsline{toc}{chapter}{Literaturverzeichnis}\n";
	ost << "% muss per Hand in d.bbl eingefuegt werden !\n";
	ost << "% nach \\begin{thebibliography}{100}\n";
	ost << endl;
	ost << "%\\begin{theindex}\n";
	ost << endl;
	ost << "%\\clearpage\n";
	ost << "%\\addcontentsline{toc}{chapter}{Index}\n";
	ost << "%\\input{apd.ind}\n";
	ost << endl;
	ost << "%\\printindex\n";
	ost << "%\\end{theindex}\n";
	ost << endl;
	ost << "}% allowdisplaybreaks\n";
	ost << endl;
	ost << "\\end{document}\n";
	ost << endl;
	ost << endl;
}




// two functions from DISCRETA1:
// adapted to use std::ostream instead of FILE pointer

void latex_interface::incma_latex_with_text_labels(
		std::ostream &fp,
		graphics::draw_incidence_structure_description *Descr,
	int v, int b,
	int V, int B, int *Vi, int *Bj,
	int *incma,
	int f_labelling_points, std::string *point_labels,
	int f_labelling_blocks, std::string *block_labels,
	int verbose_level)
// width for one box in 0.1mm
// width_10 is 1 10th of width
// example: width = 40, width_10 = 4
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "latex_interface::incma_latex_with_text_labels" << endl;
	}
	int w, h, w1, h1;
	int i, j, k, a;
	int x0, y0, x1, y1;
	int X0, Y0, X1, Y1;
	int width_8, width_5;


	string tdo_line_width;
	string line_width;


	tdo_line_width.assign(Descr->thick_lines);

	line_width.assign(Descr->thin_lines);
	// char *geo_line_width = "0.25mm";

#if 0
	if (!Descr->f_width) {
		cout << "latex_interface::incma_latex_with_text_labels please give -width <width>" << endl;
		exit(1);
	}
	if (!Descr->f_width_10) {
		cout << "latex_interface::incma_latex_with_text_labels please give -width_10 <width_10>" << endl;
		exit(1);
	}
#endif

	width_8 = Descr->width - 2 * Descr->width_10;
	width_5 = Descr->width >> 1;
	fp << "\\unitlength" << Descr->unit_length << endl;
	w = b * Descr->width;
	h = v * Descr->width;
	w1 = w;
	h1 = h;
	if (f_labelling_points) {
		w1 += 2 * Descr->width;
		}
	if (f_labelling_blocks) {
		h1 += 2 * Descr->width;
		}
	fp << "\\begin{picture}(" << w1 << "," << h1 << ")" << endl;

	// the grid:
	fp << "\\linethickness{" << tdo_line_width << "}" << endl;
	k = 0;
	for (i = -1; i < B; i++) {
		if (i >= 0) {
			a = Bj[i];
			k += a;
			}
		if (Descr->f_outline_thin) {
			if (i == -1 || i == B - 1) {
				continue;
				}
			}
		fp << "\\put(" << k * Descr->width << ",0){\\line(0,1){"
				<< h << "}}" << endl;
		}
	if (k != b) {
		cout << "incma_latex_picture: k != b" << endl;
		exit(1);
		}
	k = 0;
	for (i = -1; i < V; i++) {
		if (i >= 0) {
			a = Vi[i];
			k += a;
			}
		if (Descr->f_outline_thin) {
			if (i == -1 || i == V - 1) {
				continue;
				}
			}
		fp << "\\put(0," << h - k * Descr->width << "){\\line(1,0){"
				<< w << "}}" << endl;
		}
	if (k != v) {
		cout << "incma_latex_picture: k != v" << endl;
		exit(1);
		}

	// labeling of points:
	if (f_labelling_points) {
		for (i = 0; i < v; i++) {
			fp << "\\put(0," << h - i * Descr->width - width_5
				<< "){\\makebox(0,0)[r]{"
				<< point_labels[i] << "$\\,$}}" << endl;
			}
		}
	else {
		for (i = 0; i < v; i++) {
			fp << "\\put(0," << h - i * Descr->width - width_5
				<< "){\\makebox(0,0)[r]{}}" << endl;
			}
		}

	// labeling of blocks:
	if (f_labelling_blocks) {
		for (i = 0; i < b; i++) {
			fp << "\\put(" << i * Descr->width + width_5 << "," << h + width_5
				<< "){\\makebox(0,0)[b]{"
				<< block_labels[i] << "}}" << endl;
			}
		}
	else {
		for (i = 0; i < b; i++) {
			fp << "\\put(" << i * Descr->width + width_5 << "," << h + width_5
				<< "){\\makebox(0,0)[b]{}}" << endl;
			}
		}

	// the grid:
	fp << "\\linethickness{" << line_width << "}" << endl;
	fp << "\\multiput(0,0)(" << Descr->width << ",0){" << b + 1
			<< "}{\\line(0,1){" << h
		<< "}}" << endl;
	fp << "\\multiput(0,0)(0," << Descr->width << "){" << v + 1
			<< "}{\\line(1,0){" << w << "}}" << endl;

	// the incidence matrix itself:
	fp << "\\linethickness{" << Descr->geo_line_width << "}" << endl;
	for (i = 0; i < v; i++) {
		y0 = h - i * Descr->width;
		y1 = h - (i + 1) * Descr->width;
		Y0 = y0 - Descr->width_10;
		Y1 = y1 + Descr->width_10;
		for (j = 0; j < b; j++) {
			if (incma[i * b + j] == 0) {
				continue;
			}
			// printf("%d ", j);
			x0 = j * Descr->width;
			x1 = (j + 1) * Descr->width;
			X0 = x0 + Descr->width_10;
			X1 = x1 - Descr->width_10;
			// hor. lines:
			fp << "\\put(" << X0 << "," << Y0 << "){\\line(1,0){"
					<< width_8 << "}}" << endl;
			fp << "\\put(" << X0 << "," << Y1 << "){\\line(1,0){"
					<< width_8 << "}}" << endl;

			// vert. lines:
			fp << "\\put(" << X0 << "," << Y1 << "){\\line(0,1){"
					<< width_8 << "}}" << endl;
			fp << "\\put(" << X1 << "," << Y1 << "){\\line(0,1){"
					<< width_8 << "}}" << endl;

			}
		// printf("\n");
		}

	fp << "\\end{picture}" << endl;
	if (f_v) {
		cout << "latex_interface::incma_latex_with_text_labels done" << endl;
	}
}




void latex_interface::incma_latex(std::ostream &fp,
	int v, int b,
	int V, int B, int *Vi, int *Bj,
	int *incma,
	int verbose_level)
// used in incidence_structure::latex_it
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "latex_interface::incma_latex" << endl;
	}
	graphics::draw_incidence_structure_description *Descr;

	Descr = Orbiter->Draw_incidence_structure_description;

	if (f_v) {
		cout << "latex_interface::incma_latex before incma_latex_with_text_labels" << endl;
	}
	incma_latex_with_text_labels(fp,
			Descr,
		v, b, V, B, Vi, Bj, incma,
		FALSE /* f_labelling_points */, NULL,
		FALSE /* f_labelling_blocks */, NULL,
		verbose_level);
	if (f_v) {
		cout << "latex_interface::incma_latex after incma_latex_with_text_labels" << endl;
	}

	if (f_v) {
		cout << "latex_interface::incma_latex done" << endl;
	}
}

void latex_interface::incma_latex_with_labels(
		std::ostream &fp,
	int v, int b,
	int V, int B, int *Vi, int *Bj,
	int *row_labels_int,
	int *col_labels_int,
	int *incma,
	int verbose_level)
// used in incidence_structure::latex_it
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "latex_interface::incma_latex" << endl;
	}
	graphics::draw_incidence_structure_description *Descr;

	Descr = Orbiter->Draw_incidence_structure_description;

	std::string *point_labels;
	std::string *block_labels;


	point_labels = new string [v];
	block_labels = new string [b];

	int i, j;

	char str[1000];


	for (i = 0; i < v; i++) {
		snprintf(str, sizeof(str), "%d", row_labels_int[i]);
		point_labels[i].assign(str);
	}
	for (j = 0; j < b; j++) {
		snprintf(str, sizeof(str), "%d", col_labels_int[j]);
		block_labels[j].assign(str);
	}

	if (f_v) {
		cout << "latex_interface::incma_latex before incma_latex_with_text_labels" << endl;
	}
	incma_latex_with_text_labels(fp,
			Descr,
		v, b, V, B, Vi, Bj, incma,
		TRUE /* f_labelling_points */, point_labels,
		TRUE /* f_labelling_blocks */, block_labels,
		verbose_level);
	if (f_v) {
		cout << "latex_interface::incma_latex after incma_latex_with_text_labels" << endl;
	}

	delete [] point_labels;
	delete [] block_labels;

	if (f_v) {
		cout << "latex_interface::incma_latex done" << endl;
	}
}



void latex_interface::print_01_matrix_tex(
		std::ostream &ost,
	int *p, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) {
		cout << "\t\"";
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			}
		ost << "\"" << endl;
		}
}

void latex_interface::print_integer_matrix_tex(
		std::ostream &ost,
	int *p, int m, int n)
{
	int i, j;

	ost << "\\begin{array}{*{" << n << "}c}" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			if (j < n - 1) {
				ost << "  & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
}

void latex_interface::print_lint_matrix_tex(
		std::ostream &ost,
	long int *p, int m, int n)
{
	int i, j;

	ost << "\\begin{array}{*{" << n << "}c}" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			if (j < n - 1) {
				ost << "  & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
}

void latex_interface::print_longinteger_matrix_tex(
		std::ostream &ost,
		ring_theory::longinteger_object *p, int m, int n)
{
	int i, j;

	ost << "\\begin{array}{*{" << n << "}r}" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			if (j < n - 1) {
				ost << "  & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
}

void latex_interface::print_integer_matrix_with_labels(
		std::ostream &ost,
	int *p, int m, int n, int *row_labels, int *col_labels,
	int f_tex)
{
	int i, j;

	if (f_tex) {
		ost << "\\begin{array}{r|*{" << n << "}r}" << endl;
		}

	for (j = 0; j < n; j++) {
		if (f_tex) {
			ost << " & ";
			}
		else {
			ost << " ";
			}
		ost << col_labels[j];
		}
	if (f_tex) {
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		}
	else {
		ost << endl;
		}
	for (i = 0; i < m; i++) {
		ost << row_labels[i];
		for (j = 0; j < n; j++) {
			if (f_tex) {
				ost << " & ";
				}
			else {
				ost << " ";
				}
			ost << p[i * n + j];
			}
		if (f_tex) {
			ost << "\\\\";
			}
		ost << endl;
		}
	if (f_tex) {
		ost << "\\end{array}" << endl;
		}
}

void latex_interface::print_lint_matrix_with_labels(std::ostream &ost,
	long int *p, int m, int n, long int *row_labels, long int *col_labels,
	int f_tex)
{
	int i, j;

	if (f_tex) {
		ost << "\\begin{array}{r|*{" << n << "}r}" << endl;
		}

	for (j = 0; j < n; j++) {
		if (f_tex) {
			ost << " & ";
			}
		else {
			ost << " ";
			}
		ost << col_labels[j];
		}
	if (f_tex) {
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		}
	else {
		ost << endl;
		}
	for (i = 0; i < m; i++) {
		ost << row_labels[i];
		for (j = 0; j < n; j++) {
			if (f_tex) {
				ost << " & ";
				}
			else {
				ost << " ";
				}
			ost << p[i * n + j];
			}
		if (f_tex) {
			ost << "\\\\";
			}
		ost << endl;
		}
	if (f_tex) {
		ost << "\\end{array}" << endl;
		}
}

void latex_interface::print_integer_matrix_with_standard_labels(
		std::ostream &ost,
	int *p, int m, int n, int f_tex)
{
	print_integer_matrix_with_standard_labels_and_offset(ost,
		p, m, n, 0, 0, f_tex);

}

void latex_interface::print_lint_matrix_with_standard_labels(
		std::ostream &ost,
	long int *p, int m, int n, int f_tex)
{
	print_lint_matrix_with_standard_labels_and_offset(ost,
		p, m, n, 0, 0, f_tex);

}

void latex_interface::print_integer_matrix_with_standard_labels_and_offset(
		std::ostream &ost,
	int *p, int m, int n, int m_offset, int n_offset, int f_tex)
{
	if (f_tex) {
		print_integer_matrix_with_standard_labels_and_offset_tex(
			ost, p, m, n, m_offset, n_offset);
		}
	else {
		print_integer_matrix_with_standard_labels_and_offset_text(
			ost, p, m, n, m_offset, n_offset);
		}
}

void latex_interface::print_lint_matrix_with_standard_labels_and_offset(
		std::ostream &ost,
	long int *p, int m, int n, int m_offset, int n_offset, int f_tex)
{
	if (f_tex) {
		print_lint_matrix_with_standard_labels_and_offset_tex(
			ost, p, m, n, m_offset, n_offset);
		}
	else {
		print_lint_matrix_with_standard_labels_and_offset_text(
			ost, p, m, n, m_offset, n_offset);
		}
}

void latex_interface::print_integer_matrix_with_standard_labels_and_offset_text(
		std::ostream &ost, int *p, int m, int n,
		int m_offset, int n_offset)
{
	int i, j, w;

	w = Orbiter->Int_vec->matrix_max_log_of_entries(p, m, n);

	for (j = 0; j < w; j++) {
		ost << " ";
		}
	for (j = 0; j < n; j++) {
		ost << " " << setw(w) << n_offset + j;
		}
	ost << endl;
	for (i = 0; i < m; i++) {
		ost << setw(w) << m_offset + i;
		for (j = 0; j < n; j++) {
			ost << " " << setw(w) << p[i * n + j];
			}
		ost << endl;
		}
}

void latex_interface::print_lint_matrix_with_standard_labels_and_offset_text(
		std::ostream &ost, long int *p, int m, int n,
		int m_offset, int n_offset)
{
	int i, j, w;

	w = Orbiter->Lint_vec->matrix_max_log_of_entries(p, m, n);

	for (j = 0; j < w; j++) {
		ost << " ";
		}
	for (j = 0; j < n; j++) {
		ost << " " << setw(w) << n_offset + j;
		}
	ost << endl;
	for (i = 0; i < m; i++) {
		ost << setw(w) << m_offset + i;
		for (j = 0; j < n; j++) {
			ost << " " << setw(w) << p[i * n + j];
			}
		ost << endl;
		}
}

void latex_interface::print_integer_matrix_with_standard_labels_and_offset_tex(
		std::ostream &ost, int *p, int m, int n,
	int m_offset, int n_offset)
{
	int i, j;

	ost << "\\begin{array}{r|*{" << n << "}{r}}" << endl;

	for (j = 0; j < n; j++) {
		ost << " & " << n_offset + j;
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << m_offset + i;
		for (j = 0; j < n; j++) {
			ost << " & " << p[i * n + j];
			}
		ost << "\\\\";
		ost << endl;
		}
	ost << "\\end{array}" << endl;
}

void latex_interface::print_lint_matrix_with_standard_labels_and_offset_tex(
		std::ostream &ost, long int *p, int m, int n,
	int m_offset, int n_offset)
{
	int i, j;

	ost << "\\begin{array}{r|*{" << n << "}{r}}" << endl;

	for (j = 0; j < n; j++) {
		ost << " & " << n_offset + j;
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << m_offset + i;
		for (j = 0; j < n; j++) {
			ost << " & " << p[i * n + j];
			}
		ost << "\\\\";
		ost << endl;
		}
	ost << "\\end{array}" << endl;
}

void latex_interface::print_integer_matrix_tex_block_by_block(
		std::ostream &ost,
	int *p, int m, int n, int block_width)
{
	int i, j, I, J, nb_row_blocks, nb_col_blocks, v, w;
	int *M;

	nb_row_blocks = (m + block_width - 1) / block_width;
	nb_col_blocks = (n + block_width - 1) / block_width;
	M = NEW_int(block_width * block_width);
	for (I = 0; I < nb_row_blocks; I++) {
		for (J = 0; J < nb_col_blocks; J++) {
			ost << "$$" << endl;
			w = block_width;
			if ((J + 1) * block_width > n) {
				w = n - J * block_width;
				}
			v = block_width;
			if ((I + 1) * block_width > m) {
				v = m - I * block_width;
				}
			for (i = 0; i < v; i++) {
				for (j = 0; j < w; j++) {
					M[i * w + j] =
							p[(I * block_width + i) * n +
							  J * block_width + j];
					}
				}
			cout << "print_integer_matrix_tex_block_by_block I="
				<< I << " J=" << J << " v=" << v
				<< " w=" << w << " M=" << endl;
			Int_matrix_print(M, v, w);
			print_integer_matrix_with_standard_labels_and_offset(
				ost, M, v, w,
				I * block_width,
				J * block_width,
				TRUE /* f_tex*/);
#if 0
			ost << "\\begin{array}{*{" << w << "}{r}}" << endl;
			for (i = 0; i < block_width; i++) {
				if (I * block_width + i > m) {
					continue;
					}
				for (j = 0; j < w; j++) {
					ost << p[i * n + J * block_width + j];
					if (j < w - 1) {
						ost << "  & ";
						}
					}
				ost << "\\\\" << endl;
				}
			ost << "\\end{array}" << endl;
#endif
			ost << "$$" << endl;
			} // next J
		} // next I
	FREE_int(M);
}

void latex_interface::print_big_integer_matrix_tex(
		std::ostream &ost,
	int *p, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			}
		ost << "\\\\" << endl;
		}
}

void latex_interface::int_vec_print_as_matrix(
		std::ostream &ost,
	int *v, int len, int width, int f_tex)
{
	int *w;
	int i;

	w = NEW_int(len + width - 1);
	Int_vec_copy(v, w, len);
	for (i = 0; i < width - 1; i++) {
		w[len + i] = 0;
		}

	print_integer_matrix_with_standard_labels(ost,
		w, (len + width - 1) / width, width, f_tex);

	FREE_int(w);
}

void latex_interface::lint_vec_print_as_matrix(
		std::ostream &ost,
	long int *v, int len, int width, int f_tex)
{
	long int *w;
	int i;

	w = NEW_lint(len + width - 1);
	Lint_vec_copy(v, w, len);
	for (i = 0; i < width - 1; i++) {
		w[len + i] = 0;
		}

	print_lint_matrix_with_standard_labels(ost,
		w, (len + width - 1) / width, width, f_tex);

	FREE_lint(w);
}


void latex_interface::int_matrix_print_with_labels_and_partition(
		std::ostream &ost,
	int *p, int m, int n,
	int *row_labels, int *col_labels,
	int *row_part_first, int *row_part_len, int nb_row_parts,
	int *col_part_first, int *col_part_len, int nb_col_parts,
	void (*process_function_or_NULL)(int *p, int m, int n,
		int i, int j, int val,
		std::string &output, void *data),
	void *data,
	int f_tex)
{
	int i, j, I, J, u, v;
	string output;

	if (f_tex) {
		ost << "\\begin{array}{r|";
		for (J = 0; J < nb_col_parts; J++) {
			ost << "*{" << col_part_len[J] << "}{r}|";
			}
		ost << "}" << endl;
		}

	for (j = 0; j < n; j++) {
		if (f_tex) {
			ost << " & ";
			}
		else {
			ost << " ";
			}
		output.assign("");
		if (process_function_or_NULL) {
			(*process_function_or_NULL)(
				p, m, n, -1, j,
				col_labels[j], output, data);
			ost << output;
			}
		else {
			ost << col_labels[j];
			}
		}
	if (f_tex) {
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		}
	else {
		ost << endl;
		}
	for (I = 0; I < nb_row_parts; I++) {
		for (u = 0; u < row_part_len[I]; u++) {
			i = row_part_first[I] + u;

			output.assign("");
			if (process_function_or_NULL) {
				(*process_function_or_NULL)(
					p, m, n, i, -1,
					row_labels[i], output, data);
				ost << output;
				}
			else {
				ost << row_labels[i];
				}

			for (J = 0; J < nb_col_parts; J++) {
				for (v = 0; v < col_part_len[J]; v++) {
					j = col_part_first[J] + v;
					if (f_tex) {
						ost << " & ";
						}
					else {
						ost << " ";
						}
					output.assign("");
					if (process_function_or_NULL) {
						(*process_function_or_NULL)(
						p, m, n, i, j, p[i * n + j],
						output, data);
						ost << output;
						}
					else {
						ost << p[i * n + j];
						}
					}
				}
			if (f_tex) {
				ost << "\\\\";
				}
			ost << endl;
			}
		if (f_tex) {
			ost << "\\hline";
			}
		ost << endl;
		}
	if (f_tex) {
		ost << "\\end{array}" << endl;
		}
}

void latex_interface::lint_matrix_print_with_labels_and_partition(
		std::ostream &ost,
	long int *p, int m, int n,
	int *row_labels, int *col_labels,
	int *row_part_first, int *row_part_len, int nb_row_parts,
	int *col_part_first, int *col_part_len, int nb_col_parts,
	void (*process_function_or_NULL)(long int *p, int m, int n,
		int i, int j, int val, std::string &output, void *data),
	void *data,
	int f_tex)
{
	int i, j, I, J, u, v;
	string output;

	if (f_tex) {
		ost << "\\begin{array}{r|";
		for (J = 0; J < nb_col_parts; J++) {
			ost << "*{" << col_part_len[J] << "}{r}|";
			}
		ost << "}" << endl;
		}

	for (j = 0; j < n; j++) {
		if (f_tex) {
			ost << " & ";
			}
		else {
			ost << " ";
			}
		output.assign("");
		if (process_function_or_NULL) {
			(*process_function_or_NULL)(
				p, m, n, -1, j,
				col_labels[j], output, data);
			ost << output;
			}
		else {
			ost << col_labels[j];
			}
		}
	if (f_tex) {
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
		}
	else {
		ost << endl;
		}
	for (I = 0; I < nb_row_parts; I++) {
		for (u = 0; u < row_part_len[I]; u++) {
			i = row_part_first[I] + u;

			output.assign("");
			if (process_function_or_NULL) {
				(*process_function_or_NULL)(
					p, m, n, i, -1,
					row_labels[i], output, data);
				ost << output;
				}
			else {
				ost << row_labels[i];
				}

			for (J = 0; J < nb_col_parts; J++) {
				for (v = 0; v < col_part_len[J]; v++) {
					j = col_part_first[J] + v;
					if (f_tex) {
						ost << " & ";
						}
					else {
						ost << " ";
						}
					output.assign("");
					if (process_function_or_NULL) {
						(*process_function_or_NULL)(
						p, m, n, i, j, p[i * n + j],
						output, data);
						ost << output;
						}
					else {
						ost << p[i * n + j];
						}
					}
				}
			if (f_tex) {
				ost << "\\\\";
				}
			ost << endl;
			}
		if (f_tex) {
			ost << "\\hline";
			}
		ost << endl;
		}
	if (f_tex) {
		ost << "\\end{array}" << endl;
		}
}

void latex_interface::int_matrix_print_tex(
		std::ostream &ost, int *p, int m, int n)
{
	int i, j;

	ost << "\\begin{array}{*{" << n << "}{c}}" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			if (j < n - 1) {
				ost << " & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
}

void latex_interface::lint_matrix_print_tex(
		std::ostream &ost, long int *p, int m, int n)
{
	int i, j;

	ost << "\\begin{array}{*{" << n << "}{c}}" << endl;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			ost << p[i * n + j];
			if (j < n - 1) {
				ost << " & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
}

void latex_interface::print_cycle_tex_with_special_point_labels(
		std::ostream &ost, int *pts, int nb_pts,
		void (*point_label)(std::stringstream &sstr,
				int pt, void *data),
		void *point_label_data)
{
	int i, pt;

	ost << "(";
	for (i = 0; i < nb_pts; i++) {
		pt = pts[i];
		stringstream sstr;
		(*point_label)(sstr, pt, point_label_data);
		ost << sstr.str();
		if (i < nb_pts - 1) {
			ost << ", ";
		}
	}
	ost << ")";
}

void latex_interface::int_set_print_tex(
		std::ostream &ost, int *v, int len)
{
	int i;

	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " \\}";
}

void latex_interface::lint_set_print_tex(
		std::ostream &ost, long int *v, int len)
{
	int i;

	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " \\}";
}

void latex_interface::lint_set_print_tex_text_mode(
		std::ostream &ost, long int *v, int len)
{
	int i;

	ost << "$\\{$ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ", ";
		}
	ost << " $\\}$";
}

void latex_interface::print_type_vector_tex(
		std::ostream &ost, int *v, int len)
// v[len + 1]
{
	int i, a;
	int f_first = TRUE;


	for (i = len; i >= 0; i--) {
		a = v[i];
		//ost << "$" << a;
		if (a == 0) {
			continue;
		}
		if (f_first) {
			f_first = FALSE;
		}
		else {
			ost << ",\\,";
		}
		ost << i;
		if (a > 9) {
			ost << "^{" << a << "}";
		}
		else if (a > 1) {
			ost << "^" << a;
		}
	}
}

void latex_interface::int_set_print_masked_tex(
		std::ostream &ost,
	int *v, int len,
	const char *mask_begin,
	const char *mask_end)
{
	int i;

	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << mask_begin << v[i] << mask_end;
		if (i < len - 1)
			ost << ", ";
		}
	ost << " \\}";
}


void latex_interface::lint_set_print_masked_tex(
		std::ostream &ost,
	long int *v, int len,
	const char *mask_begin,
	const char *mask_end)
{
	int i;

	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << mask_begin << v[i] << mask_end;
		if (i < len - 1)
			ost << ", ";
		}
	ost << " \\}";
}


void latex_interface::int_set_print_tex_for_inline_text(
		std::ostream &ost,
	int *v, int len)
{
	int i;

	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ",$ $";
		}
	ost << " \\}";
}

void latex_interface::lint_set_print_tex_for_inline_text(
		std::ostream &ost,
	long int *v, int len)
{
	int i;

	ost << "\\{ ";
	for (i = 0; i < len; i++) {
		ost << v[i];
		if (i < len - 1)
			ost << ",$ $";
		}
	ost << " \\}";
}

void latex_interface::latexable_string(
		std::stringstream &str,
		const char *p, int max_len, int line_skip)
{
	const char *q = p;
	char c;
	int len = 0;

	while ((c = *q) != 0) {
		if (c == '\t') {
			str << "\\>";
			len += 2;
		}
		else if (c == ' ') {
			str << "\\ ";
			len++;
		}
		else if (c == '\\') {
			str << "\\symbol{92}";
			len++;
		}
		else if (c == '\'') {
			str << "\\symbol{19}";
			len++;
		}
		else if (c == ',') {
			str << "\\symbol{44}";
			len++;
		}
		else if (c == '!') {
			str << "\\symbol{33}";
			len++;
		}
		else if (c == '"') {
			str << "\\symbol{34}";
			len++;
		}
		else if (c == '.') {
			str << ".";
			//str << "\\symbol{46}";
			len++;
		}
		else if (c == '-') {
			str << "\\symbol{45}";
			len++;
		}
		else if (c == '#') {
			str << "\\symbol{35}";
			len++;
		}
		else if (c == '$') {
			str << "\\symbol{36}";
			len++;
		}
		else if (c == '&') {
			str << "\\symbol{38}";
			len++;
		}
		else if (c == '~') {
			str << "\\symbol{126}";
			len++;
		}
		else if (c == '_') {
			str << "\\_";
			len++;
		}
		else if (c == '^') {
			str << "\\symbol{94}";
			len++;
		}
		else if (c == '%') {
			str << "\\symbol{37}";
			len++;
		}
		else if (c == '{') {
			str << "\\symbol{123}";
			len++;
		}
		else if (c == '}') {
			str << "\\symbol{125}";
			len++;
		}
		else if (c == '\n') {
			str << "\\\\[" << line_skip << "pt]\n";
			len = 0;
			}
		else {
			str << c;
			len++;
		}
		if (len > max_len) {
			str << "\n";
			len = 0;
		}
		q++;
	}
}

void latex_interface::print_row_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	long int *row_class_size, int nb_row_classes,
	long int *col_class_size, int nb_col_classes,
	long int *row_scheme)
{
	int i, j;

	ost << "%{\\renewcommand{\\arraycolsep}{1pt}" << endl;
	if (f_enter_math_mode) {
		ost << "$$" << endl;
		}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << "\\rightarrow ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		ost << setw(6) << col_class_size[j];
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		ost << setw(6) << row_class_size[i];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << row_scheme[i * nb_col_classes + j];
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "$$" << endl;
		}
	ost << "%}" << endl;
}

void latex_interface::print_column_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	long int *row_class_size, int nb_row_classes,
	long int *col_class_size, int nb_col_classes,
	long int *col_scheme)
{
	int i, j;

	ost << "%{\\renewcommand{\\arraycolsep}{1pt}" << endl;
	if (f_enter_math_mode) {
		ost << "$$" << endl;
		}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << "\\downarrow ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		ost << setw(6) << col_class_size[j];
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		ost << setw(6) << row_class_size[i];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << col_scheme[i * nb_col_classes + j];
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "$$" << endl;
		}
	ost << "%}" << endl;
}


}}}


