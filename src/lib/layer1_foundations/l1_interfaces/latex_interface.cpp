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
namespace l1_interfaces {


latex_interface::latex_interface()
{

}

latex_interface::~latex_interface()
{

}

void latex_interface::head_easy(
		std::ostream& ost)
{
	std::string dummy;

	dummy.assign("");

	head(ost,
		false /* f_book */,
		false /* f_title */,
		dummy, dummy,
		false /*f_toc */,
		false /* f_landscape */,
		false /* f_12pt */,
		false /* f_enlarged_page */,
		true /* f_pagenumbers */,
		dummy /* extras_for_preamble */);

}

void latex_interface::head_easy_and_enlarged(
		std::ostream& ost)
{
	std::string dummy;

	dummy.assign("");

	head(ost,
		false /* f_book */,
		false /* f_title */,
		dummy, dummy,
		false /*f_toc */,
		false /* f_landscape */,
		false /* f_12pt */,
		true /* f_enlarged_page */,
		true /* f_pagenumbers */,
		dummy /* extras_for_preamble */);

}

void latex_interface::head_easy_with_extras_in_the_praeamble(
		std::ostream& ost, std::string &extras)
{
	std::string dummy;

	dummy.assign("");

	head(ost,
		false /* f_book */,
		false /* f_title */,
		dummy, dummy,
		false /*f_toc */,
		false /* f_landscape */,
		false /* f_12pt */,
		false /* f_enlarged_page */,
		true /* f_pagenumbers */,
		extras /* extras_for_preamble */);

}

void latex_interface::head_easy_sideways(
		std::ostream& ost)
{
	std::string dummy;

	dummy.assign("");

	head(ost, false /* f_book */,
		false /* f_title */,
		dummy, dummy,
		false /*f_toc */,
		true /* f_landscape */,
		false /* f_12pt */,
		false /* f_enlarged_page */,
		true /* f_pagenumbers */,
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
	if (f_book) {
		ost << "book";
	}
	else {
		ost << "article";
	}
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
		//ost << "\\usepackage[a4paper]{geometry}" << endl;
		//ost << "\\usepackage[a3paper]{geometry}" << endl;
		//ost << "\\usepackage[a2paper]{geometry}" << endl;
		ost << "\\usepackage[a1paper]{geometry}" << endl;
		//ost << "\\usepackage[a0paper]{geometry}" << endl;

		//ost << "\\usepackage{fullpage}" << endl;
		//ost << "\\usepackage[top=1in,bottom=0.2in,right=1in,left=1in]{geometry}" << endl; // A Betten 2/7/2021
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
	ost << "%\\parindent=0pt" << endl;
	ost << endl;
	//ost << "\\renewcommand{\\baselinestretch}{1.5}" << endl;
	ost << endl;


#if 0
	if (f_enlarged_page) {
		ost << "\\hoffset -2cm" << endl;
		ost << "\\voffset -1cm" << endl;
		ost << "\\topmargin 0.0cm" << endl;
		if (f_landscape) {
			ost << "\\textheight=18cm" << endl;
			ost << "\\textwidth=23cm" << endl;
		}
		else {
			ost << "\\textheight=23cm" << endl;
			ost << "\\textwidth=18cm" << endl;
		}
	}
	else {
		ost << "\\hoffset -0.7cm" << endl;
		ost << "%\\voffset 0cm" << endl;
		ost << endl;
		ost << "%\\oddsidemargin=15pt" << endl;
		ost << endl;
		ost << "%\\oddsidemargin 0pt" << endl;
		ost << "%\\evensidemargin 0pt" << endl;
		ost << "%\\topmargin 0pt" << endl;
		ost << endl;
#if 1
		if (f_landscape) {
			ost << "\\textwidth = 20cm" << endl;
			ost << "\\textheight= 17cm" << endl;
		}
		else {
			ost << "\\textwidth = 17cm" << endl;
			ost << "\\textheight= 21cm" << endl;
		}
		ost << endl;
#endif
	}
#endif


	ost << "%\\topmargin=0pt" << endl;
	ost << "%\\headsep=18pt" << endl;
	ost << "%\\footskip=45pt" << endl;
	ost << "%\\mathsurround=1pt" << endl;
	ost << "%\\evensidemargin=0pt" << endl;
	ost << "%\\oddsidemargin=15pt" << endl;
	ost << endl;

	ost << "%\\setlength{\\textheight}{\\baselineskip*41+\\topskip}" << endl;
	ost << endl;


	ost << "\\newcommand{\\sectionline}{" << endl;
	ost << "   \\nointerlineskip \\vspace{\\baselineskip}" << endl;
	ost << "   \\hspace{\\fill}\\rule{0.9\\linewidth}{1.7pt}\\hspace{\\fill}" << endl;
	ost << "   \\par\\nointerlineskip \\vspace{\\baselineskip}" << endl;
	ost << "   }" << endl;

	ost << "\\newcommand\\setTBstruts{\\def\\T{\\rule{0pt}{2.6ex}}%" << endl;
	ost << "\\def\\B{\\rule[-1.2ex]{0pt}{0pt}}}" << endl;

	ost << "\\newcommand{\\ans}[1]{\\\\{\\bf ANSWER}: {#1}}" << endl;
	ost << "\\newcommand{\\Aut}{{\\rm Aut}}" << endl;
	ost << "\\newcommand{\\Sym}{{\\rm Sym}}" << endl;
	ost << "\\newcommand{\\sFix}{{\\cal Fix}}" << endl;
	ost << "\\newcommand{\\sOrbits}{{\\cal Orbits}}" << endl;
	//ost << "\\newcommand{\\sFix}{{\\mathscr Fix}}" << endl;
	//ost << "\\newcommand{\\sOrbits}{{\\mathscr Orbits}}" << endl;
	ost << "\\newcommand{\\Stab}{{\\rm Stab}}" << endl;
	ost << "\\newcommand{\\Fix}{{\\rm Fix}}" << endl;
	ost << "\\newcommand{\\fix}{{\\rm fix}}" << endl;
	ost << "\\newcommand{\\Orbits}{{\\rm Orbits}}" << endl;
	ost << "\\newcommand{\\PG}{{\\rm PG}}" << endl;
	ost << "\\newcommand{\\AG}{{\\rm AG}}" << endl;
	ost << "\\newcommand{\\SQS}{{\\rm SQS}}" << endl;
	ost << "\\newcommand{\\STS}{{\\rm STS}}" << endl;
	//ost << "\\newcommand{\\Sp}{{\\rm Sp}}" << endl;
	ost << "\\newcommand{\\PSL}{{\\rm PSL}}" << endl;
	ost << "\\newcommand{\\PGL}{{\\rm PGL}}" << endl;
	ost << "\\newcommand{\\PSSL}{{\\rm P\\Sigma L}}" << endl;
	ost << "\\newcommand{\\PGGL}{{\\rm P\\Gamma L}}" << endl;
	ost << "\\newcommand{\\SL}{{\\rm SL}}" << endl;
	ost << "\\newcommand{\\GL}{{\\rm GL}}" << endl;
	ost << "\\newcommand{\\SSL}{{\\rm \\Sigma L}}" << endl;
	ost << "\\newcommand{\\GGL}{{\\rm \\Gamma L}}" << endl;
	ost << "\\newcommand{\\ASL}{{\\rm ASL}}" << endl;
	ost << "\\newcommand{\\AGL}{{\\rm AGL}}" << endl;
	ost << "\\newcommand{\\ASSL}{{\\rm A\\Sigma L}}" << endl;
	ost << "\\newcommand{\\AGGL}{{\\rm A\\Gamma L}}" << endl;
	ost << "\\newcommand{\\PSU}{{\\rm PSU}}" << endl;
	ost << "\\newcommand{\\HS}{{\\rm HS}}" << endl;
	ost << "\\newcommand{\\Hol}{{\\rm Hol}}" << endl;
	ost << "\\newcommand{\\SO}{{\\rm SO}}" << endl;
	ost << "\\newcommand{\\ASO}{{\\rm ASO}}" << endl;

	ost << "\\newcommand{\\la}{\\langle}" << endl;
	ost << "\\newcommand{\\ra}{\\rangle}" << endl;


	ost << "\\newcommand{\\cA}{{\\cal A}}" << endl;
	ost << "\\newcommand{\\cB}{{\\cal B}}" << endl;
	ost << "\\newcommand{\\cC}{{\\cal C}}" << endl;
	ost << "\\newcommand{\\cD}{{\\cal D}}" << endl;
	ost << "\\newcommand{\\cE}{{\\cal E}}" << endl;
	ost << "\\newcommand{\\cF}{{\\cal F}}" << endl;
	ost << "\\newcommand{\\cG}{{\\cal G}}" << endl;
	ost << "\\newcommand{\\cH}{{\\cal H}}" << endl;
	ost << "\\newcommand{\\cI}{{\\cal I}}" << endl;
	ost << "\\newcommand{\\cJ}{{\\cal J}}" << endl;
	ost << "\\newcommand{\\cK}{{\\cal K}}" << endl;
	ost << "\\newcommand{\\cL}{{\\cal L}}" << endl;
	ost << "\\newcommand{\\cM}{{\\cal M}}" << endl;
	ost << "\\newcommand{\\cN}{{\\cal N}}" << endl;
	ost << "\\newcommand{\\cO}{{\\cal O}}" << endl;
	ost << "\\newcommand{\\cP}{{\\cal P}}" << endl;
	ost << "\\newcommand{\\cQ}{{\\cal Q}}" << endl;
	ost << "\\newcommand{\\cR}{{\\cal R}}" << endl;
	ost << "\\newcommand{\\cS}{{\\cal S}}" << endl;
	ost << "\\newcommand{\\cT}{{\\cal T}}" << endl;
	ost << "\\newcommand{\\cU}{{\\cal U}}" << endl;
	ost << "\\newcommand{\\cV}{{\\cal V}}" << endl;
	ost << "\\newcommand{\\cW}{{\\cal W}}" << endl;
	ost << "\\newcommand{\\cX}{{\\cal X}}" << endl;
	ost << "\\newcommand{\\cY}{{\\cal Y}}" << endl;
	ost << "\\newcommand{\\cZ}{{\\cal Z}}" << endl;

	ost << "\\newcommand{\\rmA}{{\\rm A}}" << endl;
	ost << "\\newcommand{\\rmB}{{\\rm B}}" << endl;
	ost << "\\newcommand{\\rmC}{{\\rm C}}" << endl;
	ost << "\\newcommand{\\rmD}{{\\rm D}}" << endl;
	ost << "\\newcommand{\\rmE}{{\\rm E}}" << endl;
	ost << "\\newcommand{\\rmF}{{\\rm F}}" << endl;
	ost << "\\newcommand{\\rmG}{{\\rm G}}" << endl;
	ost << "\\newcommand{\\rmH}{{\\rm H}}" << endl;
	ost << "\\newcommand{\\rmI}{{\\rm I}}" << endl;
	ost << "\\newcommand{\\rmJ}{{\\rm J}}" << endl;
	ost << "\\newcommand{\\rmK}{{\\rm K}}" << endl;
	ost << "\\newcommand{\\rmL}{{\\rm L}}" << endl;
	ost << "\\newcommand{\\rmM}{{\\rm M}}" << endl;
	ost << "\\newcommand{\\rmN}{{\\rm N}}" << endl;
	ost << "\\newcommand{\\rmO}{{\\rm O}}" << endl;
	ost << "\\newcommand{\\rmP}{{\\rm P}}" << endl;
	ost << "\\newcommand{\\rmQ}{{\\rm Q}}" << endl;
	ost << "\\newcommand{\\rmR}{{\\rm R}}" << endl;
	ost << "\\newcommand{\\rmS}{{\\rm S}}" << endl;
	ost << "\\newcommand{\\rmT}{{\\rm T}}" << endl;
	ost << "\\newcommand{\\rmU}{{\\rm U}}" << endl;
	ost << "\\newcommand{\\rmV}{{\\rm V}}" << endl;
	ost << "\\newcommand{\\rmW}{{\\rm W}}" << endl;
	ost << "\\newcommand{\\rmX}{{\\rm X}}" << endl;
	ost << "\\newcommand{\\rmY}{{\\rm Y}}" << endl;
	ost << "\\newcommand{\\rmZ}{{\\rm Z}}" << endl;

	ost << "\\newcommand{\\bA}{{\\bf A}}" << endl;
	ost << "\\newcommand{\\bB}{{\\bf B}}" << endl;
	ost << "\\newcommand{\\bC}{{\\bf C}}" << endl;
	ost << "\\newcommand{\\bD}{{\\bf D}}" << endl;
	ost << "\\newcommand{\\bE}{{\\bf E}}" << endl;
	ost << "\\newcommand{\\bF}{{\\bf F}}" << endl;
	ost << "\\newcommand{\\bG}{{\\bf G}}" << endl;
	ost << "\\newcommand{\\bH}{{\\bf H}}" << endl;
	ost << "\\newcommand{\\bI}{{\\bf I}}" << endl;
	ost << "\\newcommand{\\bJ}{{\\bf J}}" << endl;
	ost << "\\newcommand{\\bK}{{\\bf K}}" << endl;
	ost << "\\newcommand{\\bL}{{\\bf L}}" << endl;
	ost << "\\newcommand{\\bM}{{\\bf M}}" << endl;
	ost << "\\newcommand{\\bN}{{\\bf N}}" << endl;
	ost << "\\newcommand{\\bO}{{\\bf O}}" << endl;
	ost << "\\newcommand{\\bP}{{\\bf P}}" << endl;
	ost << "\\newcommand{\\bQ}{{\\bf Q}}" << endl;
	ost << "\\newcommand{\\bR}{{\\bf R}}" << endl;
	ost << "\\newcommand{\\bS}{{\\bf S}}" << endl;
	ost << "\\newcommand{\\bT}{{\\bf T}}" << endl;
	ost << "\\newcommand{\\bU}{{\\bf U}}" << endl;
	ost << "\\newcommand{\\bV}{{\\bf V}}" << endl;
	ost << "\\newcommand{\\bW}{{\\bf W}}" << endl;
	ost << "\\newcommand{\\bX}{{\\bf X}}" << endl;
	ost << "\\newcommand{\\bY}{{\\bf Y}}" << endl;
	ost << "\\newcommand{\\bZ}{{\\bf Z}}" << endl;

#if 0
	ost << "\\newcommand{\\sA}{{\\mathscr A}}" << endl;
	ost << "\\newcommand{\\sB}{{\\mathscr B}}" << endl;
	ost << "\\newcommand{\\sC}{{\\mathscr C}}" << endl;
	ost << "\\newcommand{\\sD}{{\\mathscr D}}" << endl;
	ost << "\\newcommand{\\sE}{{\\mathscr E}}" << endl;
	ost << "\\newcommand{\\sF}{{\\mathscr F}}" << endl;
	ost << "\\newcommand{\\sG}{{\\mathscr G}}" << endl;
	ost << "\\newcommand{\\sH}{{\\mathscr H}}" << endl;
	ost << "\\newcommand{\\sI}{{\\mathscr I}}" << endl;
	ost << "\\newcommand{\\sJ}{{\\mathscr J}}" << endl;
	ost << "\\newcommand{\\sK}{{\\mathscr K}}" << endl;
	ost << "\\newcommand{\\sL}{{\\mathscr L}}" << endl;
	ost << "\\newcommand{\\sM}{{\\mathscr M}}" << endl;
	ost << "\\newcommand{\\sN}{{\\mathscr N}}" << endl;
	ost << "\\newcommand{\\sO}{{\\mathscr O}}" << endl;
	ost << "\\newcommand{\\sP}{{\\mathscr P}}" << endl;
	ost << "\\newcommand{\\sQ}{{\\mathscr Q}}" << endl;
	ost << "\\newcommand{\\sR}{{\\mathscr R}}" << endl;
	ost << "\\newcommand{\\sS}{{\\mathscr S}}" << endl;
	ost << "\\newcommand{\\sT}{{\\mathscr T}}" << endl;
	ost << "\\newcommand{\\sU}{{\\mathscr U}}" << endl;
	ost << "\\newcommand{\\sV}{{\\mathscr V}}" << endl;
	ost << "\\newcommand{\\sW}{{\\mathscr W}}" << endl;
	ost << "\\newcommand{\\sX}{{\\mathscr X}}" << endl;
	ost << "\\newcommand{\\sY}{{\\mathscr Y}}" << endl;
	ost << "\\newcommand{\\sZ}{{\\mathscr Z}}" << endl;
#else
	ost << "\\newcommand{\\sA}{{\\cal A}}" << endl;
	ost << "\\newcommand{\\sB}{{\\cal B}}" << endl;
	ost << "\\newcommand{\\sC}{{\\cal C}}" << endl;
	ost << "\\newcommand{\\sD}{{\\cal D}}" << endl;
	ost << "\\newcommand{\\sE}{{\\cal E}}" << endl;
	ost << "\\newcommand{\\sF}{{\\cal F}}" << endl;
	ost << "\\newcommand{\\sG}{{\\cal G}}" << endl;
	ost << "\\newcommand{\\sH}{{\\cal H}}" << endl;
	ost << "\\newcommand{\\sI}{{\\cal I}}" << endl;
	ost << "\\newcommand{\\sJ}{{\\cal J}}" << endl;
	ost << "\\newcommand{\\sK}{{\\cal K}}" << endl;
	ost << "\\newcommand{\\sL}{{\\cal L}}" << endl;
	ost << "\\newcommand{\\sM}{{\\cal M}}" << endl;
	ost << "\\newcommand{\\sN}{{\\cal N}}" << endl;
	ost << "\\newcommand{\\sO}{{\\cal O}}" << endl;
	ost << "\\newcommand{\\sP}{{\\cal P}}" << endl;
	ost << "\\newcommand{\\sQ}{{\\cal Q}}" << endl;
	ost << "\\newcommand{\\sR}{{\\cal R}}" << endl;
	ost << "\\newcommand{\\sS}{{\\cal S}}" << endl;
	ost << "\\newcommand{\\sT}{{\\cal T}}" << endl;
	ost << "\\newcommand{\\sU}{{\\cal U}}" << endl;
	ost << "\\newcommand{\\sV}{{\\cal V}}" << endl;
	ost << "\\newcommand{\\sW}{{\\cal W}}" << endl;
	ost << "\\newcommand{\\sX}{{\\cal X}}" << endl;
	ost << "\\newcommand{\\sY}{{\\cal Y}}" << endl;
	ost << "\\newcommand{\\sZ}{{\\cal Z}}" << endl;
#endif

	ost << "\\newcommand{\\frakA}{{\\mathfrak A}}" << endl;
	ost << "\\newcommand{\\frakB}{{\\mathfrak B}}" << endl;
	ost << "\\newcommand{\\frakC}{{\\mathfrak C}}" << endl;
	ost << "\\newcommand{\\frakD}{{\\mathfrak D}}" << endl;
	ost << "\\newcommand{\\frakE}{{\\mathfrak E}}" << endl;
	ost << "\\newcommand{\\frakF}{{\\mathfrak F}}" << endl;
	ost << "\\newcommand{\\frakG}{{\\mathfrak G}}" << endl;
	ost << "\\newcommand{\\frakH}{{\\mathfrak H}}" << endl;
	ost << "\\newcommand{\\frakI}{{\\mathfrak I}}" << endl;
	ost << "\\newcommand{\\frakJ}{{\\mathfrak J}}" << endl;
	ost << "\\newcommand{\\frakK}{{\\mathfrak K}}" << endl;
	ost << "\\newcommand{\\frakL}{{\\mathfrak L}}" << endl;
	ost << "\\newcommand{\\frakM}{{\\mathfrak M}}" << endl;
	ost << "\\newcommand{\\frakN}{{\\mathfrak N}}" << endl;
	ost << "\\newcommand{\\frakO}{{\\mathfrak O}}" << endl;
	ost << "\\newcommand{\\frakP}{{\\mathfrak P}}" << endl;
	ost << "\\newcommand{\\frakQ}{{\\mathfrak Q}}" << endl;
	ost << "\\newcommand{\\frakR}{{\\mathfrak R}}" << endl;
	ost << "\\newcommand{\\frakS}{{\\mathfrak S}}" << endl;
	ost << "\\newcommand{\\frakT}{{\\mathfrak T}}" << endl;
	ost << "\\newcommand{\\frakU}{{\\mathfrak U}}" << endl;
	ost << "\\newcommand{\\frakV}{{\\mathfrak V}}" << endl;
	ost << "\\newcommand{\\frakW}{{\\mathfrak W}}" << endl;
	ost << "\\newcommand{\\frakX}{{\\mathfrak X}}" << endl;
	ost << "\\newcommand{\\frakY}{{\\mathfrak Y}}" << endl;
	ost << "\\newcommand{\\frakZ}{{\\mathfrak Z}}" << endl;

	ost << "\\newcommand{\\fraka}{{\\mathfrak a}}" << endl;
	ost << "\\newcommand{\\frakb}{{\\mathfrak b}}" << endl;
	ost << "\\newcommand{\\frakc}{{\\mathfrak c}}" << endl;
	ost << "\\newcommand{\\frakd}{{\\mathfrak d}}" << endl;
	ost << "\\newcommand{\\frake}{{\\mathfrak e}}" << endl;
	ost << "\\newcommand{\\frakf}{{\\mathfrak f}}" << endl;
	ost << "\\newcommand{\\frakg}{{\\mathfrak g}}" << endl;
	ost << "\\newcommand{\\frakh}{{\\mathfrak h}}" << endl;
	ost << "\\newcommand{\\fraki}{{\\mathfrak i}}" << endl;
	ost << "\\newcommand{\\frakj}{{\\mathfrak j}}" << endl;
	ost << "\\newcommand{\\frakk}{{\\mathfrak k}}" << endl;
	ost << "\\newcommand{\\frakl}{{\\mathfrak l}}" << endl;
	ost << "\\newcommand{\\frakm}{{\\mathfrak m}}" << endl;
	ost << "\\newcommand{\\frakn}{{\\mathfrak n}}" << endl;
	ost << "\\newcommand{\\frako}{{\\mathfrak o}}" << endl;
	ost << "\\newcommand{\\frakp}{{\\mathfrak p}}" << endl;
	ost << "\\newcommand{\\frakq}{{\\mathfrak q}}" << endl;
	ost << "\\newcommand{\\frakr}{{\\mathfrak r}}" << endl;
	ost << "\\newcommand{\\fraks}{{\\mathfrak s}}" << endl;
	ost << "\\newcommand{\\frakt}{{\\mathfrak t}}" << endl;
	ost << "\\newcommand{\\fraku}{{\\mathfrak u}}" << endl;
	ost << "\\newcommand{\\frakv}{{\\mathfrak v}}" << endl;
	ost << "\\newcommand{\\frakw}{{\\mathfrak w}}" << endl;
	ost << "\\newcommand{\\frakx}{{\\mathfrak x}}" << endl;
	ost << "\\newcommand{\\fraky}{{\\mathfrak y}}" << endl;
	ost << "\\newcommand{\\frakz}{{\\mathfrak z}}" << endl;


	ost << "\\newcommand{\\Tetra}{{\\mathfrak Tetra}}" << endl;
	ost << "\\newcommand{\\Cube}{{\\mathfrak Cube}}" << endl;
	ost << "\\newcommand{\\Octa}{{\\mathfrak Octa}}" << endl;
	ost << "\\newcommand{\\Dode}{{\\mathfrak Dode}}" << endl;
	ost << "\\newcommand{\\Ico}{{\\mathfrak Ico}}" << endl;

	ost << "\\newcommand{\\bbF}{{\\mathbb F}}" << endl;
	ost << "\\newcommand{\\bbQ}{{\\mathbb Q}}" << endl;
	ost << "\\newcommand{\\bbC}{{\\mathbb C}}" << endl;
	ost << "\\newcommand{\\bbR}{{\\mathbb R}}" << endl;

	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\makeindex" << endl;
	ost << endl;
	ost << "\\begin{document} " << endl;
	ost << "\\setTBstruts" << endl;
	ost << endl;
	ost << "\\bibliographystyle{plain}" << endl;
	if (!f_pagenumbers) {
		ost << "\\pagestyle{empty}" << endl;
	}
	ost << "%\\large" << endl;
	ost << endl;
	ost << "{\\allowdisplaybreaks%" << endl;
	ost << endl;
	ost << endl;
	ost << endl;
	ost << endl;
	ost << "%\\makeindex" << endl;
	ost << endl;
	ost << "%\\renewcommand{\\labelenumi}{(\\roman{enumi})}" << endl;
	ost << endl;

	if (f_title) {
		ost << "\\title{" << title << "}" << endl;
		ost << "\\author{" << author << "}%end author" << endl;
		ost << "%\\date{}" << endl;
		ost << "\\maketitle%" << endl;
	}
	ost << "%\\pagenumbering{roman}" << endl;
	ost << "%\\thispagestyle{empty}" << endl;
	if (f_toc) {
		ost << "\\tableofcontents" << endl;
	}
	ost << "%\\input et.tex%" << endl;
	ost << "%\\thispagestyle{empty}%\\phantom{page2}%\\clearpage%" << endl;
	ost << "%\\addcontentsline{toc}{chapter}{Inhaltsverzeichnis}%" << endl;
	ost << "%\\tableofcontents" << endl;
	ost << "%\\listofsymbols" << endl;
	if (f_toc){
		ost << "\\clearpage" << endl;
		ost << endl;
	}
	ost << "%\\pagenumbering{arabic}" << endl;
	ost << "%\\pagenumbering{roman}" << endl;
	ost << endl;
	ost << endl;
	ost << endl;
}


void latex_interface::foot(
		std::ostream& ost)
{
	ost << endl;
	ost << endl;
	ost << "%\\bibliographystyle{gerplain}% wird oben eingestellt" << endl;
	ost << "%\\addcontentsline{toc}{section}{References}" << endl;
	ost << "%\\bibliography{../MY_BIBLIOGRAPHY/anton}" << endl;
	ost << "% ACHTUNG: nicht vergessen:" << endl;
	ost << "% die Zeile" << endl;
	ost << "%\\addcontentsline{toc}{chapter}{Literaturverzeichnis}" << endl;
	ost << "% muss per Hand in d.bbl eingefuegt werden !" << endl;
	ost << "% nach \\begin{thebibliography}{100}" << endl;
	ost << endl;
	ost << "%\\begin{theindex}" << endl;
	ost << endl;
	ost << "%\\clearpage" << endl;
	ost << "%\\addcontentsline{toc}{chapter}{Index}" << endl;
	ost << "%\\input{apd.ind}" << endl;
	ost << endl;
	ost << "%\\printindex" << endl;
	ost << "%\\end{theindex}" << endl;
	ost << endl;
	ost << "}% allowdisplaybreaks" << endl;
	ost << endl;
	ost << "\\end{document}" << endl;
	ost << endl;
	ost << endl;
}




// two functions from DISCRETA1:

void latex_interface::incma_latex_with_text_labels(
		std::ostream &fp,
		graphics::draw_incidence_structure_description *Descr,
	int v, int b,
	int V, int B, int *Vi, int *Bj,
	int *incma,
	int f_labelling_points, std::string *point_labels,
	int f_labelling_blocks, std::string *block_labels,
	int verbose_level)
// output incidence geometry as a latex picture
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
		cout << "latex_interface::incma_latex_with_text_labels "
				"please give -width <width>" << endl;
		exit(1);
	}
	if (!Descr->f_width_10) {
		cout << "latex_interface::incma_latex_with_text_labels "
				"please give -width_10 <width_10>" << endl;
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

	// thick vertical lines according to the partition:

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

	// thick horizontal lines according to the partition:

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
	}

	fp << "\\end{picture}" << endl;
	if (f_v) {
		cout << "latex_interface::incma_latex_with_text_labels done" << endl;
	}
}




void latex_interface::incma_latex(
		std::ostream &fp,
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

	Descr = orbiter_kernel_system::Orbiter->Draw_incidence_structure_description;

	if (f_v) {
		cout << "latex_interface::incma_latex "
				"before incma_latex_with_text_labels" << endl;
	}
	incma_latex_with_text_labels(fp,
			Descr,
		v, b, V, B, Vi, Bj, incma,
		false /* f_labelling_points */, NULL,
		false /* f_labelling_blocks */, NULL,
		verbose_level);
	if (f_v) {
		cout << "latex_interface::incma_latex "
				"after incma_latex_with_text_labels" << endl;
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

	Descr = orbiter_kernel_system::Orbiter->Draw_incidence_structure_description;

	std::string *point_labels;
	std::string *block_labels;


	point_labels = new string [v];
	block_labels = new string [b];

	int i, j;


	for (i = 0; i < v; i++) {
		point_labels[i] = std::to_string(row_labels_int[i]);
	}
	for (j = 0; j < b; j++) {
		block_labels[j] = std::to_string(col_labels_int[j]);
	}

	if (f_v) {
		cout << "latex_interface::incma_latex "
				"before incma_latex_with_text_labels" << endl;
	}
	incma_latex_with_text_labels(fp,
			Descr,
		v, b, V, B, Vi, Bj, incma,
		true /* f_labelling_points */, point_labels,
		true /* f_labelling_blocks */, block_labels,
		verbose_level);
	if (f_v) {
		cout << "latex_interface::incma_latex "
				"after incma_latex_with_text_labels" << endl;
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
	print_integer_matrix_with_standard_labels_and_offset(
			ost,
		p, m, n, 0, 0, f_tex);

}

void latex_interface::print_lint_matrix_with_standard_labels(
		std::ostream &ost,
	long int *p, int m, int n, int f_tex)
{
	print_lint_matrix_with_standard_labels_and_offset(
			ost,
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

	w = orbiter_kernel_system::Orbiter->Int_vec->matrix_max_log_of_entries(p, m, n);

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

	w = orbiter_kernel_system::Orbiter->Lint_vec->matrix_max_log_of_entries(p, m, n);

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
				true /* f_tex*/);
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

	if (len < width) {

		print_integer_matrix_with_standard_labels(ost,
			v, 1, len, f_tex);

	}
	else {
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
		if (i < len - 1) {
			ost << ", ";
		}
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
		if (i < len - 1) {
			ost << ", ";
		}
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
		if (i < len - 1) {
			ost << ", ";
		}
	}
	ost << " $\\}$";
}

void latex_interface::print_type_vector_tex(
		std::ostream &ost, int *v, int len)
// v[len + 1]
{
	int i, a;
	int f_first = true;


	for (i = len; i >= 0; i--) {
		a = v[i];
		//ost << "$" << a;
		if (a == 0) {
			continue;
		}
		if (f_first) {
			f_first = false;
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
		if (i < len - 1) {
			ost << ", ";
		}
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
		if (i < len - 1) {
			ost << ", ";
		}
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
		if (i < len - 1) {
			ost << ",$ $";
		}
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
		if (i < len - 1) {
			ost << ",$ $";
		}
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
			str << "\\\\[" << line_skip << "pt];" << endl;
			len = 0;
			}
		else {
			str << c;
			len++;
		}
		if (len > max_len) {
			str << endl;
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

void latex_interface::report_matrix(
		std::string &fname,
		std::string &title,
		std::string &author,
		std::string &extra_praeamble,
	int *M, int nb_rows, int nb_cols)
{
	ofstream ost(fname);

	head(ost,
			false /* f_book*/,
			true /* f_title */,
			title, author,
			false /* f_toc */,
			false /* f_landscape */,
			true /* f_12pt */,
			true /* f_enlarged_page */,
			true /* f_pagenumbers */,
			extra_praeamble /* extra_praeamble */);



	ost << "$$" << endl;
	ost << "\\left[" << endl;
	int_matrix_print_tex(
			ost, M, nb_rows, nb_cols);
	ost << "\\right]" << endl;
	ost << "$$" << endl;

	Int_vec_print_fully(
			ost, M, nb_rows * nb_cols);
	ost << "\\\\" << endl;



	foot(ost);

}


void latex_interface::report_matrix_longinteger(
		std::string &fname,
		std::string &title,
		std::string &author,
		std::string &extra_praeamble,
		ring_theory::longinteger_object *M, int nb_rows, int nb_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "latex_interface::report_matrix_longinteger" << endl;
	}

	{
		ofstream ost(fname);

		head(ost,
				false /* f_book*/,
				true /* f_title */,
				title, author,
				false /* f_toc */,
				false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		//int i, j;

		ost << "$$" << endl;

		print_longinteger_matrix_tex(
				ost,
				M, nb_rows, nb_cols);

	#if 0
		ost << "\\begin{array}{*{" << nb_cols << "}{r}}" << endl;
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				ost << M[i * nb_cols + j];
				if (j < nb_cols - 1) {
					ost << " & ";
				}
			}
			ost << "\\\\" << endl;
		}
		ost << "\\end{array}" << endl;
	#endif

		ost << "$$" << endl;


		foot(ost);
	}

	if (f_v) {
		orbiter_kernel_system::file_io Fio;

		cout << "latex_interface::report_matrix_longinteger "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "latex_interface::report_matrix_longinteger done" << endl;
	}
}

void latex_interface::print_decomposition_matrix(
		std::ostream &ost,
		int m, int n,
		std::string &top_left_entry,
		std::string *cols_labels,
		std::string *row_labels,
		std::string *entries,
		int f_enter_math_mode)
{
	int i, j;

	ost << "%{\\renewcommand{\\arraycolsep}{1pt}" << endl;
	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
	}
	ost << "\\begin{array}{r|*{" << n << "}{r}}" << endl;
	ost << top_left_entry;
	for (j = 0; j < n; j++) {
		ost << " & " << cols_labels[j];
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < m; i++) {
		ost << row_labels[i];
		for (j = 0; j < n; j++) {
			ost << " & " << entries[i * n + j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
	}
	ost << "%}" << endl;

}

}}}


