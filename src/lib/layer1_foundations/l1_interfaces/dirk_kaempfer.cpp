/*
 * dirk_kaempfer.cpp
 *
 *  Created on: Aug 27, 2024
 *      Author: betten
 *
 *      based on tdo3.c by Dirk Kaempfer.
 */



#include "foundations.h"

#include <sstream>

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace l1_interfaces {

/* Hans Dirk Kaempfer */

/* Anton Betten 1/6/2018: eliminated compiler warnings about C  comments
 * and about printf argument strings with long int parameters */


/* USE_STOUT 1  eingebaut, damit man in einen file leiten kann und den
 *              automatisch zippen kann. Zeile auskommentieren, wenn man alte
 * Version haben will*/
/*#define __GNUC__*/
//#define __GCC__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define USE_STDOUT 1
//#include <iostream>



/******************************************************************
 ******************************************************************
 ** DEFINE
 **
 ******************************************************************
 ******************************************************************/

#define MAX_VARANZ  100
#define MAX(a,b)  a>b ? a : b
#define MIN(a,b)  a<b ? a : b

#define DEBUGGINGMODE 0
#if DEBUGGINGMODE > 2
  #define DBG             fprintf
  #define DBG1(A)         fprintf(OUTDAT, A) ;fflush(OUTDAT);
  #define DBG2(A,B)       fprintf(OUTDAT, A, B) ;fflush(OUTDAT);
  #define DBG3(A,B,C)     fprintf(OUTDAT, A,B,C) ;fflush(OUTDAT);
  #define DBG4(A,B,C,D)   fprintf(OUTDAT, A,B,C,D) ;fflush(OUTDAT);
  #define DBG5(A,B,C,D,E)   fprintf(OUTDAT, A,B,C,D,E) ;fflush(OUTDAT);
  #define DBG6(A,B,C,D,E,F)   fprintf(OUTDAT, A,B,C,D,E,F) ;fflush(OUTDAT);
  #define DBG8(A,B,C,D,E,F,G,H)  fprintf(OUTDAT, A,B,C,D,E,F,G,H) ;fflush(OUTDAT);
#else
  #define DBG    if (1) {} else fprintf
  #define DBG1   if (1) {} else printf
  #define DBG2   if (1) {} else printf
  #define DBG3   if (1) {} else printf
  #define DBG4   if (1) {} else printf
  #define DBG5   if (1) {} else printf
  #define DBG8   if (1) {} else printf
#endif
#if DEBUGGINGMODE > 1
  #define DBG_F(A)        A
#else
  #define DBG_F(A)
#endif
#if DEBUGGINGMODE > 0
  #define DBG2_(A)          if (lDbg) fprintf(OUTDAT,A)
  #define DBG2_1(A)         if (lDbg) fprintf(OUTDAT, A)
  #define DBG2_2(A,B)       if (lDbg) fprintf(OUTDAT, A, B)
  #define DBG2_3(A,B,C)     if (lDbg) fprintf(OUTDAT, A,B,C)
  #define DBG2_4(A,B,C,D)   if (lDbg) fprintf(OUTDAT, A,B,C,D)
  #define DBG2_5(A,B,C,D,E)   if (lDbg) fprintf(OUTDAT, A,B,C,D,E)
  #define DBG2_8(A,B,C,D,E,F,G,H)  if (lDbg) fprintf(OUTDAT, A,B,C,D,E,F,G,H)
  #define DBG2_F(A)        if (lDbg) A
#else
  #define DBG2_    if (1) {} else printf
  #define DBG2_1   if (1) {} else printf
  #define DBG2_2   if (1) {} else printf
  #define DBG2_3   if (1) {} else printf
  #define DBG2_4   if (1) {} else printf
  #define DBG2_5   if (1) {} else printf
  #define DBG2_8   if (1) {} else printf
  #define DBG2_F(A)
#endif


#define TIMEMODE 0
#if TIMEMODE
   #define TM(A) A
#else
   #define TM(A)
#endif

#ifndef TRUE
# define TRUE    1
# define FALSE   0
#endif



/******************************************************************
 ******************************************************************
 ** TYPEDEF
 **
 ******************************************************************
 ******************************************************************/

typedef short  SHORT;
typedef int    INT;
typedef long   LONG;
typedef void   VOID;
typedef INT    VECTOR[MAX_VARANZ];
typedef LONG   GLEICHUNG[MAX_VARANZ];
typedef VECTOR MATRIX[MAX_VARANZ];
typedef INT    BOOL;

typedef struct {
   INT iAnzahl;
   INT iLaenge;
}  GERADENTYP;

typedef struct {
   INT        iPunkteAnzahl;
   INT        iGeradenAnzahl;
   INT        iGeradenTypAnzahl;
   BOOL       bGeradenAnzahlProTypVorgegeben;
   GERADENTYP agt[MAX_VARANZ];
}  GERADENFALL;

struct ST_HTDO {
   INT    iNeuAnzahl,          /* Gesamtanzahl der Exemplare der neu zu   */
                               /* spezifizierenden Klasse (Punkte,Geraden)*/
          iAltAnzahl,          /* Gesamtanzahl der Exemplare der zuletzt  */
                               /* spezifizierten Klasse (Punkte,Geraden)  */
          iNeuTypAnzahlVorher, /* Die Anzahl der bereits unterscheidbaren */
                               /* Typen der neu zu spezifizierenden       */
                               /* Klasse (Punkte,Geraden) (NeuKlasse)     */
          iAltTypAnzahlVorher, /* Die Anzahl der bereits unterscheidbaren */
                               /* Typen der zuletzt spezifizierten        */
                               /* Klasse (Punkte,Geraden) (AltKlasse)     */
                               /* vor der letzen Spezifikation            */
          iAltTypAnzahl,       /* Die Anzahl der bereits unterscheidbaren */
                               /* Typen der zuletzt spezifizierten        */
                               /* Klasse (Punkte,Geraden) (AltKlasse)     */
          iNeuTypAnzahl,       /* Die Anzahl der unterscheidbaren Typen   */
                               /* der NeuKlasse in der matHtdo            */
          iStufe;
   VECTOR vecNeuVorher,        /* Die Exemplar-Anzahlen der                */
                               /* (iNeuTypAnzahlVorher) Typen der NeuKlasse*/
                               /* vor der letzen Spezifikation            */
          vecAltVorher,        /* Die Exemplar-Anzahlen der                */
                               /* (iAltTypAnzahlVorher) Typen der AltKlasse*/
                               /* vor der letzen Spezifikation            */
          vecNeuTypVorherEnde, /* enthaelt zu jedem NeuTypVorher den  */
                               /* Index des letzten Exemplars         */
          vecNeuTypVorherAnf,  /* enthaelt zu jedem NeuTypVorher den  */
                               /* Index des ersten Exemplars         */
          vecAltTypVorherEnde, /* enthaelt zu jedem AltTypVorher den  */
                               /* Index des letzten Exemplars         */
          vecAlt,              /* Die Exemplar-Anzahlen der (iAltTypAnzahl)*/
                               /* Typen der AltKlasse                      */
          vecNeu;              /* Die Exemplar-Anzahlen der (iNeuTypAnzahl)*/
                               /* Typen der NeuKlasse in der matHtdo       */
   MATRIX matInz,              /* Hier wird jedes Exemplar der NeuKlasse  */
                               /* einzeln behandelt (Arbeitsmatrix)       */
          matHtdo;             /* Eine fertige matInz wird zu einer       */
                               /* matHtdo durch Normalisierung (gleiche   */
                               /* Exemplare werden zusammengefasst)       */
   struct ST_HTDO* phtdoAlt;   /* Pointer auf HTDO der nchstkleineren Stufe*/
};
typedef struct ST_HTDO HTDO;

typedef struct {
   INT      iMinimum,
            iZweier;
   VECTOR   vecFahnen,
            vecPktVerbindungen;
   MATRIX   matGerSchneidungen;
   BOOL     bOrdnen;
}  HTDO1_PARMS;

typedef struct {
   INT      iNeuExemplar,            /* Aktuell zu spezifizierendes         */
                                     /* Exemplar der NeuKlasse              */
            iNeuTypVorher,           /* Typ, zu dem iNeuExemplar bei der    */
                                     /* letzten Spezifikation gehrte       */
            iNeuTypVorherIndex,      /* Position des NeuTypVorher im        */
                                     /* Indexvector                         */
            iAltTyp,                 /* Typ der AltKlasse, zu dem gerade die*/
                                     /* Anzahl der Inzidenzen gesucht wird  */
            iAltTypVorher,           /* Typ, zu dem iAltTyp vor dessen      */
                                     /* letzter Spezifikation gehrte       */
            iMinimum;                /* MindestAnzahl der Inzidenzen        */
                                     /* zwischen iNeuExemplar und iAltTyp   */
   VECTOR   vecNeuTypVorherIndex;    /* nach Anzahl geordnete Reihenfolge   */
                                     /* der NeuTypenVorher,                 */
                                     /* beginnend mit der kleinsten Anzahl  */
   MATRIX   matAltTypInzidenzen,     /* enthlt zu jedem AltTyp die Anzahl  */
                                     /* der Inzidenzen mit den AltTypen     */
                                     /* gráeren Indexes                    */
            matAltTypFahnen,         /* enthlt zu jedem NeuTypVorher die   */
                                     /* Anzahlen der Fahnen der AltTypen    */
            matNeuTypInzidenzen,     /* enthlt zu jedem NeuExemplar die    */
                                     /* Anzahl der Inzidenzen mit den       */
                                     /* NeuTypenVorher                      */
            matNeuTypFahnenVorher;   /* enthlt zu jedem NeuExemplar die    */
                                     /* Anzahl der Fahnen der AltTypenVorher*/
   BOOL     bOrdnen;                 /* gibt an, ob die Inzidenzen des aktu-*/
                                     /* ellen NeuExemplars noch kleiner sein*/
                                     /* mssen als beim vorigen Exemplar    */
}  HTDO_PARMS;



/******************************************************************
 ******************************************************************
 ** STATIC
 **
 ******************************************************************
 ******************************************************************/

#define _FNSIZE 256
static FILE *INDAT, *OUTDAT, *TESTFALLIN, *TESTFALLOUT;  /* EIN- UND AUSGABEDATEIEN */
static char INDAT_NAME[_FNSIZE];
static char OUTDAT_NAME[_FNSIZE];
static char TESTFALLIN_NAME[_FNSIZE];
static char TESTFALLOUT_NAME[_FNSIZE];

static GLEICHUNG A;
static VECTOR    vecLoesung, vecTestfall;
static INT       iMaxStufe, iTestfallAnzahl;
static LONG      lSumme, lLoesungsAnzahl;
static INT       aiStartTestfall[20],iVariablenAnzahl, iTestfallStufe;
static char      sStartTestfall[80];
static LONG      lLenStartTestfall,lTestfallAnzahl;
static BOOL      bBearbeiteTestfall,
                 bStartTestfallErreicht, bDruckeHtdos,
                 bHtdosProtokollieren, bDruckeZweier;
static MATRIX    matMaxfit;

static long      lMaxTime,lErhoeheTime,lNiedrigTime,lInitTime,lHtdoTime;
static long      lMaxAnz,lErhoeheAnz,lNiedrigAnz,lInitAnz;
static clock_t   cl;
static long      lDbg;

/******************************************************************
 ******************************************************************
 ** PREDECLARATIONS
 **
 ******************************************************************
 ******************************************************************/
static void AusgabeTestfall(HTDO *phtdo);




/******************************************************************
 ******************************************************************
 ** oeffneDateien
 **
 ******************************************************************
 ******************************************************************/
static void oeffneDateien()
{
  /* Oeffnen der Dateien */
  OUTDAT      = NULL;
  INDAT       = NULL;
  TESTFALLIN  = NULL;
  TESTFALLOUT = NULL;
  strcpy(OUTDAT_NAME, "ausgabe.dat");
#ifdef USE_STDOUT
    OUTDAT = stdout;
#else
    OUTDAT = fopen(OUTDAT_NAME, "w");
#endif
  strcpy(INDAT_NAME, "eingabe.dat");
    INDAT = fopen(INDAT_NAME, "r");

};  /* oeffneDateien */




/******************************************************************
 ******************************************************************
 ** schliesseDateien
 **
 ******************************************************************
 ******************************************************************/
static void schliesseDateien()
{
   if (INDAT != NULL)
      fclose(INDAT);
   if (OUTDAT != NULL)
      fclose(OUTDAT);
   if (TESTFALLIN != NULL)
      fclose(TESTFALLIN);
   if (TESTFALLOUT != NULL)
      fclose(TESTFALLOUT);

} /* schliesseDateien */


/******************************************************************
 ******************************************************************
 ** LesenGeraden
 ******************************************************************
 ******************************************************************/
static void LesenGeraden(GERADENFALL *pgf)
{
  char        sPuffer[80];
  INT         i, j;
  LONG        lLiesAnzahlen;

  /******************************************************************
   * Suchen von "Wenn nein"
   ******************************************************************/
  do {
    fgets(sPuffer,sizeof(sPuffer), INDAT);
  } while ( memcmp(sPuffer,"Wenn nein",9) ); /* enddo */

  /******************************************************************
   * Lesen PunktAnzahl
   ******************************************************************/
  do {
    fgets(sPuffer,sizeof(sPuffer), INDAT);
  } while ( '0'>sPuffer[0] || '9'<sPuffer[0] ); /* enddo */
  sscanf(sPuffer,"%d", &pgf->iPunkteAnzahl);
  fprintf(OUTDAT,"Anzahl der Punkte: %d \n", pgf->iPunkteAnzahl);
  fflush(OUTDAT);

  /******************************************************************
   * Lesen GeradenTypAnzahl
   ******************************************************************/
  fgets(sPuffer,sizeof(sPuffer), INDAT);
  sscanf(sPuffer,"%d", &pgf->iGeradenTypAnzahl);
  fprintf(OUTDAT,"Anzahl der Geradentypen: %d \n", pgf->iGeradenTypAnzahl);

  /******************************************************************
   * Lesen, ob die Anzahlen der Geraden vorgegeben werden sollen *
   ******************************************************************/
  do {
    fgets(sPuffer,sizeof(sPuffer), INDAT);
  } while ( '0'>sPuffer[0] || '9'<sPuffer[0] ); /* enddo */
  sscanf(sPuffer,"%ld ", &lLiesAnzahlen);
  DBG2("Vorgabe der Anzahlen der Geraden: %ld\n", lLiesAnzahlen);
  pgf->bGeradenAnzahlProTypVorgegeben = (lLiesAnzahlen != 0);

  /******************************************************************
   * Lesen der Geradenlaengen und u.U. ihrer Anzahlen *
   ******************************************************************/
  for (i = 0; i < pgf->iGeradenTypAnzahl; i++) {
    do {
      fgets(sPuffer,sizeof(sPuffer), INDAT);
    } while ( '0'>sPuffer[0] || '9'<sPuffer[0] ); /* enddo */
    sscanf(sPuffer,"%d %d ", &pgf->agt[i].iLaenge, &pgf->agt[i].iAnzahl);
    fprintf(OUTDAT,"Anzahl der Punkte auf Geradentyp %d:  %d\n", i+1,
                   pgf->agt[i].iLaenge);
  }

  /******************************************************************
   * Sortieren der Geraden (Bubblesort): Die lngsten Geraden zuerst
   ******************************************************************/
  for (j=pgf->iGeradenTypAnzahl -1; j>=1 ; j--) {
    for (i=0; i<j ; i++ ) {
       if ( pgf->agt[i].iLaenge < pgf->agt[i+1].iLaenge ) {
          /* Tausch der beiden GeradenTypen */
          GERADENTYP gtTausch       = pgf->agt[i];
          pgf->agt[i]               = pgf->agt[i+1];
          pgf->agt[i+1]             = gtTausch;
        }  /* endif */
    } /* endfor */
  } /* endfor */
  DBG2("Nach Tausch agt[0]: %d\n",pgf->agt[0].iLaenge);

  /******************************************************************
   * Daten-Ueberpruefung bei Vorgabe der GeradenAnzahlen            *
   ******************************************************************/
  if (lLiesAnzahlen>0) {
     long lInzidenzenAnzahl = pgf->iPunkteAnzahl * (pgf->iPunkteAnzahl-1);
     for (i=0; i<pgf->iGeradenTypAnzahl; i++) {
        lInzidenzenAnzahl -= pgf->agt[i].iLaenge * (pgf->agt[i].iLaenge-1) * pgf->agt[i].iAnzahl;
     } /* endfor */
     pgf->bGeradenAnzahlProTypVorgegeben = (lInzidenzenAnzahl == 0);
     DBG2("lInzidenzanzahl: %ld \n",lInzidenzenAnzahl);
  };  /* endif */

fflush(OUTDAT);
return;
} /* LesenGeraden */



/******************************************************************
 ******************************************************************
 ** gfLesenEingabe
 ** TRUE : Die GeradenTypAnzahlen wurden vorgegeben
 ** FALSE: Die GeradenTypAnzahlen wurden nicht vorgegeben
 ******************************************************************
 ******************************************************************/
static GERADENFALL gfLesenEingabe()
{
  BOOL        b  = FALSE;
  char        sPuffer[80];
  INT         i, j;
  LONG        l,lLiesAnzahlen;
  GERADENFALL gf;

  /******************************************************************
   * Lesen ob ein bestehender Testfall bearbeitet werden soll
   ******************************************************************/
  do {
    fgets(sPuffer,sizeof(sPuffer), INDAT);
  } while ( '0'>sPuffer[0] || '9'<sPuffer[0] ); /* enddo */
  sscanf(sPuffer,"%ld", &l);
  bBearbeiteTestfall = (l == 1);
  fprintf(OUTDAT,"Testfall einlesen: %d \n", bBearbeiteTestfall);
  fflush(OUTDAT);

  /******************************************************************
   * u.U. Einlesen der Eingabedatei der Testfaelle
   ******************************************************************/
  if (bBearbeiteTestfall) {
     fgets(sPuffer,sizeof(sPuffer), INDAT);
     sscanf(sPuffer,"%s", TESTFALLIN_NAME);
     fprintf(OUTDAT,"Testfall-Datei:%s\n",TESTFALLIN_NAME);
     TESTFALLIN = fopen(TESTFALLIN_NAME, "r");

     /******************************************************************
      * Einlesen des Start-Testfalles                                  *
      ******************************************************************/
     do {
        fgets(sPuffer,sizeof(sPuffer), INDAT);
     } while ('0'>sPuffer[0] || '9'<sPuffer[0]); /* enddo */
     sscanf(sPuffer,"%s",sStartTestfall);
     fprintf(OUTDAT,"Start-Testfall:%s\n",sStartTestfall);
     fflush(OUTDAT);
     lLenStartTestfall = strlen(sStartTestfall);

     /******************************************************************
      * Lesen Anzahl zu bearbeitender Testflle
      ******************************************************************/
     do {
       fgets(sPuffer,sizeof(sPuffer), INDAT);
     } while ( '0'>sPuffer[0] || '9'<sPuffer[0] ); /* enddo */
     sscanf(sPuffer,"%d", &iTestfallAnzahl);
     fprintf(OUTDAT,"Anzahl der Testfaelle: %d \n", iTestfallAnzahl);
     fflush(OUTDAT);
  } else {
     /******************************************************************
      * u.U. Einlesen der Punkte und Geraden
      ******************************************************************/
     LesenGeraden(&gf);
  } /* endif */

  /******************************************************************
   * Weiterlesen bis zur Zeile "Steuerungsangaben:"
   ******************************************************************/
  do {
    fgets(sPuffer,sizeof(sPuffer), INDAT);
    DBG2("%80.80s\n",sPuffer);
  } while ( memcmp(sPuffer,"Steuerungsangaben:",18)); /* enddo */

  /******************************************************************
   * Einlesen der maximalen Berechnungssstufe der TDOs              *
   ******************************************************************/
  do {
    fgets(sPuffer,sizeof(sPuffer), INDAT);
  } while ('0'>sPuffer[0] || '9'<sPuffer[0]); /* enddo */
  DBG2("%80.80s\n",sPuffer);
  sscanf(sPuffer,"%d", &iMaxStufe);
  fprintf(OUTDAT,"Maximale Tiefe: %d  \n",iMaxStufe);fflush(OUTDAT);

  /******************************************************************
   * Einlesen des Schalters,
   * ob die Testfaelle wieder einlesbar sein sollen
   * und der TESTFALLOUT-Datei
   ******************************************************************/
  do {
    fgets(sPuffer,sizeof(sPuffer), INDAT);
  } while ( '0'>sPuffer[0] || '9'<sPuffer[0] ); /* enddo */
  sscanf(sPuffer,"%ld ", &l);
  DBG2("Protokollieren von HTDOs: %ld\n", l);
  bHtdosProtokollieren = (1 == l);
  if (bHtdosProtokollieren) {
    fgets(sPuffer,sizeof(sPuffer), INDAT);
    sscanf(sPuffer,"%s", TESTFALLOUT_NAME);
    TESTFALLOUT = fopen(TESTFALLOUT_NAME, "w");
  } /* endif */

  /******************************************************************
   * Einlesen des Schalters, ob HTDOs gedruckt werden sollen
   ******************************************************************/
  do {
    fgets(sPuffer,sizeof(sPuffer), INDAT);
  } while ( '0'>sPuffer[0] || '9'<sPuffer[0] ); /* enddo */
  sscanf(sPuffer,"%ld ", &l);
  DBG2("Drucken von HTDOs: %ld\n", l);
  bDruckeHtdos = (1 == l);

  /******************************************************************
   * Einlesen des Schalters, ob die Typen der 2-Geraden gedruckt
   * werden sollen
   ******************************************************************/
  do {
    fgets(sPuffer,sizeof(sPuffer), INDAT);
  } while ( '0'>sPuffer[0] || '9'<sPuffer[0] ); /* enddo */
  sscanf(sPuffer,"%ld ", &l);
  bDruckeZweier = (1 == l);
  DBG2("Drucken von 2-Geraden: %d\n", bDruckeZweier);
  fprintf(OUTDAT,"Ende LesenEingabe \n");fflush(OUTDAT);

  return gf;
}  /* end bLesenEingabe */






/******************************************************************
 ******************************************************************
 ** BildeGleichung
 ** Die Summe der Verbindungen durch die Einzelgeraden muss der Anzahl
 ** der Punktepaare entsprechen.
 ******************************************************************
 ******************************************************************/
static VOID BildeGleichung(GERADENFALL * pgf)
{
   INT i;

   DBG1("BildeGleichung \n");

   /******************************************************************
    * A[0] = PunktAnzahl ber 2 = Anzahl Verbindungen auf den Punkten
    ******************************************************************/
   A[0] = pgf->iPunkteAnzahl * (pgf->iPunkteAnzahl -1 ) / 2;
   DBG3("A[%d]=%ld ",0,A[0]);

   /******************************************************************
    * A[i] = GeradenLaenge[i] ber 2
    *      = Anzahl Verbindungen einer Gerade des Typs i
    ******************************************************************/
   for ( i = 1; i <= pgf->iGeradenTypAnzahl; i++) {
      A[i] = pgf->agt[i-1].iLaenge * (pgf->agt[i-1].iLaenge -1) /2;
      DBG3("A[%d]=%ld ",i,A[i]);
   }  /* endfor */
   DBG1("\n");

   /******************************************************************
    * Initialisieren von vecLoesung
    ******************************************************************/
   lSumme = A[0];
   iVariablenAnzahl = pgf->iGeradenTypAnzahl;
   for ( i = 1; i <= iVariablenAnzahl; ++i)
     vecLoesung[i] = 0;

} /* BildeGleichung */





/******************************************************************
 ******************************************************************
 ** bGibNaechsteGleichungsloesung()
 ** TRUE  = es gibt noch eine Lsung
 ** FALSE = es gibt keine weitere Lsung
 ******************************************************************
 ******************************************************************/
static BOOL bGibNaechsteGleichungsloesung(GERADENFALL* pgf)
{
   INT         i,j;
   BOOL        bLoesbar = TRUE;

   DBG1("GibNaechsteGleichungsLoesung \n");

   if (lSumme == 0) {
      /******************************************************************
       * Dann ist dies nicht der erste Aufruf und vecLoesung ist gefllt
       * Suchen der letzten erhhbaren Variablen aus vecLoesung
       ******************************************************************/
      j = iVariablenAnzahl;
      while (lSumme < A[j] && j > 1) {
         lSumme += vecLoesung[j] * A[j];
         vecLoesung[j] = 0;
         j--;
      } /* endwhile */
      if (lSumme < A[j]) {  /*  also j=1 */
         bLoesbar = FALSE;
      } else {
         lSumme -= A[j];
         vecLoesung[j]++;
      } /* endelse */
   } /* endelse */

   /******************************************************************
    * Suchen der nchsten Lsung
    ******************************************************************/
   while (bLoesbar && lSumme > 0) {
      if (lSumme % A[iVariablenAnzahl] == 0) {
         vecLoesung[iVariablenAnzahl] = lSumme / A[iVariablenAnzahl];
         lSumme = 0;
      } else {
         j = iVariablenAnzahl - 1;
         while (lSumme < A[j] && j > 1) {
            lSumme += vecLoesung[j] * A[j];
            vecLoesung[j] = 0;
            j--;
         }  /* endwhile */
         if (lSumme < A[j]) {   /* also j == 1 */
            bLoesbar = FALSE;
         } else {
            lSumme -= A[j];
            vecLoesung[j]++;
         }  /* endif */
      }  /* endif */
   }  /* endwhile */

   /******************************************************************
    * bertragen der Loesung in den Geradenfall
    ******************************************************************/
   for ( i = 1; i <= iVariablenAnzahl; i++)
      pgf->agt[i-1].iAnzahl = vecLoesung[i];

   return bLoesbar;
}





/******************************************************************
 ******************************************************************
 ** bBraunbedingungErfuellt
 ** Es muá |P| >=  |G1|+|G2|+...+|Gn| - (n ber 2)  sein.
 ** TRUE  = Braunbedingung ist erfllt, Geometrie mglich
 ** FALSE = Braunbedingung ist nicht erfllt, Geometrie unmglich
 ******************************************************************
 ******************************************************************/
static BOOL bBraunbedingungErfuellt(GERADENFALL * pgf)
{
   INT iSumme  = 0;  /* wird zur Summe der Geradenlaengen in obiger Formel */
   INT n       = 0;  /* wird das n von oben */
   INT j       = 0;
   GERADENTYP * pgt = pgf->agt;

   DBG1( "BraunbedingungErfuellt \n");


   for ( j=0; (n + pgt->iAnzahl < pgt->iLaenge) && (j < pgf->iGeradenTypAnzahl); j++,++pgt) {
      /* alle iAnzahl iLaenge-Geraden bentigen noch weitere Punkte */
      iSumme = iSumme + pgt->iLaenge * pgt->iAnzahl;
      n      = n + pgt->iAnzahl;
   } /* endfor */

   if (j < pgf->iGeradenTypAnzahl) {
      /* jetzt ist ( pgt->iAnzahl >= pgt->iLaenge - n ) */
      /* nur noch (pgt->iLaenge-n) Geraden bentigen noch weitere Punkte */
      if ( n < pgt->iLaenge) {
         iSumme = iSumme + pgt->iLaenge * (pgt->iLaenge - n);
         n = pgt->iLaenge;
      }
   } /* endif */

   DBG4("Punkte:%d Summe:%d n:%d \n",pgf->iPunkteAnzahl,iSumme,n);

   return (pgf->iPunkteAnzahl  >= (iSumme - (n * (n-1) / 2)) );

} /* bBraunbedingungErfuellt */






/******************************************************************
 ******************************************************************
 ** AusgabeGeradenwechsel
 ******************************************************************
 ******************************************************************/
static VOID AusgabeGeradenwechsel(GERADENFALL * pgf)
{
   INT i;
   GERADENTYP * pgt= pgf->agt;

   fprintf(OUTDAT, "\nVersuch einer Geometrie auf %d Punkten mit:\n",
                    pgf->iPunkteAnzahl);

   for ( i = 0; i < pgf->iGeradenTypAnzahl; ++i,++pgt) {
      if (pgt->iAnzahl > 0)
         fprintf(OUTDAT, "%d Gerade(n) der Laenge %d\n",
                         pgt->iAnzahl, pgt->iLaenge );
      pgf->iGeradenAnzahl += pgt->iAnzahl;
   }
   fprintf(OUTDAT, "\n");
   fflush(OUTDAT);
} /* AusgabeGeradenwechsel */




/******************************************************************
 ******************************************************************
 ** BerechneZweier
 ******************************************************************
 ******************************************************************/
static INT iBerechneZweier( HTDO *phtdo, BOOL bIsTdo)
{
INT i,j,k;
INT iVerb, iZweier = 0;

if ( 1 == phtdo->iStufe % 2) {
   /* AltTyp == Punkte */
   for (i=0; i < phtdo->iAltTypAnzahl ; ++i) {
      for ( j=i; j < phtdo->iAltTypAnzahl; ++j) {
         /* iVerb ist die Anzahl der noch offenen Verbindungen zwischen */
         /* den Punkttypen i und j */
         if (i == j) {
            iVerb = phtdo->vecAlt[i] *(phtdo->vecAlt[i]-1) / 2;
         } else {
            iVerb = phtdo->vecAlt[i] * phtdo->vecAlt[j];
         } /* endif */
         for (k=0; k < phtdo->iNeuTypAnzahl; ++k) {
            if (i == j) {
               iVerb -= phtdo->matHtdo[k][i] * (phtdo->matHtdo[k][i] -1) /2 *phtdo->vecNeu[k];
            } else {
               iVerb -= phtdo->matHtdo[k][i] * phtdo->matHtdo[k][j] * phtdo->vecNeu[k];
            } /* endif */
         } /* endfor */
         if (0 < iVerb) {
            memset(phtdo->matHtdo[phtdo->iNeuTypAnzahl + iZweier],0,sizeof(VECTOR));
            if (i == j) {
               phtdo->matHtdo[phtdo->iNeuTypAnzahl + iZweier][i] = 2;
            } else {
               phtdo->matHtdo[phtdo->iNeuTypAnzahl + iZweier][i] = 1;
               phtdo->matHtdo[phtdo->iNeuTypAnzahl + iZweier][j] = 1;
            } /* endif */
            phtdo->vecNeu[phtdo->iNeuTypAnzahl + iZweier] = iVerb;
            ++iZweier;
         }
      } /* endfor */
   } /* endfor */
} else {
   if (phtdo->iStufe == 2) {
   } else {
      /* NeuTyp == Punkte */
      HTDO * phtdoAlt = phtdo->phtdoAlt;
      INT iAnzahlNTVi = 0;
      INT iAnzahlNTVj = 0;

      for (i=0; i < phtdo->iNeuTypAnzahlVorher ; ++i) {
         iAnzahlNTVi = phtdoAlt->vecAlt[i];
         for (j=i; j < phtdo->iNeuTypAnzahlVorher ; ++j) {
            iAnzahlNTVj = phtdoAlt->vecAlt[j];

            /* Berechne die Anzahl der Verbindungen zwischen NeuTypVorher i und j */
            if (i == j) {
               iVerb = iAnzahlNTVi * (iAnzahlNTVj -1) /2;
            } else {
               iVerb = iAnzahlNTVi * iAnzahlNTVj;
            } /* endif */
            for (k=0; k < phtdoAlt->iNeuTypAnzahl; ++k) {
               if (i == j) {
                  iVerb -= phtdoAlt->matHtdo[k][i] * (phtdoAlt->matHtdo[k][i] -1) /2 *phtdoAlt->vecNeu[k];
               } else {
                  iVerb -= phtdoAlt->matHtdo[k][i] * phtdoAlt->matHtdo[k][j] * phtdoAlt->vecNeu[k];
               } /* endif */
            } /* endfor */

            /* iVerb enthlt jetzt die Anzahl der noch offenen Verbindungen*/
            /* zwischen NeutypVorher i und j */
            if (iVerb > 0) {
               INT k1, k2, iVerb2, iSum;
               memset(phtdo->matHtdo[phtdo->iNeuTypAnzahl + iZweier],0,sizeof(VECTOR));

               /* Berechnung der NeuTypen des NeuTypVorher i */
               for (k1 = phtdo->vecNeuTypVorherAnf[i]; k1 <= phtdo->vecNeuTypVorherEnde[i]; k1 = iSum) {
                  if (i == j) {
                     iVerb2 = iAnzahlNTVj -1;
                  } else {
                     iVerb2 = iAnzahlNTVj;
                  } /* endif */
                  for (k2 = 0; k2 < phtdo->iAltTypAnzahl; ++k2) {
                     if (i == j) {
                        iVerb2 -= phtdo->matInz[k1][k2] * (phtdoAlt->matHtdo[k2][j] -1);
                     } else {
                        iVerb2 -= phtdo->matInz[k1][k2] * phtdoAlt->matHtdo[k2][j];
                     } /* endif */
                  } /* endfor */

                  /* iVerb2 enthaelt jetzt die Anzahl der noch Offenen Verbindungen */
                  /* zwischen dem NeuExemplar k1 (vom NeuTypVorher i) und den Punkten des */
                  /* NeuTypVorher j                                                 */

                  /* Jetzt muss der NeuTyp des Exemplars k1 ermittelt werden */
                  /* k2 enthlt den NeuTyp */
                  for (k2 = 0, iSum = phtdo->vecNeu[0]; iSum <= k1; ++k2, iSum += phtdo->vecNeu[k2]) {
                  } /* endfor */

                  phtdo->matHtdo[k2][phtdo->iAltTypAnzahl + iZweier] = iVerb2;


               } /* endfor */

               /* Berechnung der NeuTypen des NeuTypVorher j */
               for (k1 = phtdo->vecNeuTypVorherAnf[j]; k1 <= phtdo->vecNeuTypVorherEnde[j]; k1 = iSum) {
                  if (i == j) {
                     iVerb2 = iAnzahlNTVi -1;
                  } else {
                     iVerb2 = iAnzahlNTVi;
                  } /* endif */
                  for (k2 = 0; k2 < phtdo->iAltTypAnzahl; ++k2) {
                     if (i == j) {
                        iVerb2 -= phtdo->matInz[k1][k2] * (phtdoAlt->matHtdo[k2][i] -1);
                     } else {
                        iVerb2 -= phtdo->matInz[k1][k2] * phtdoAlt->matHtdo[k2][i];
                     } /* endif */
                  } /* endfor */

                  /* iVerb2 enthaelt jetzt die Anzahl der noch Offenen Verbindungen */
                  /* zwischen dem NeuExemplar k1 (vom NeuTypVorher j) und den Punkten des */
                  /* NeuTypVorher i                                                 */

                  /* Jetzt muss der NeuTyp des Exemplars k1 ermittelt werden */
                  /* k2 enthlt den NeuTyp */
                  for (k2 = 0, iSum = phtdo->vecNeu[0]; iSum <= k1; ++k2, iSum += phtdo->vecNeu[k2]) {
                  } /* endfor */

                  phtdo->matHtdo[k2][phtdo->iAltTypAnzahl + iZweier] = iVerb2;


               } /* endfor */


               phtdo->vecAlt[phtdo->iAltTypAnzahl + iZweier] = iVerb;
               ++iZweier;
            } /* endif */
         } /* endfor */
      } /* endfor */
   } /* endif */
} /* endif */

return iZweier;
}

/******************************************************************
 ******************************************************************
 ** Ausgabe
 ******************************************************************
 ******************************************************************/
static VOID Ausgabe( HTDO *phtdo, BOOL bIsTdo)
{
   INT    i,
          iGeradenAnzahl,
          iPunkteAnzahl,
          iNeuTypZaehler,
          iAltTypZaehler;
   INT  * piGeradenZaehler,
        * piPunkteZaehler;
   INT  * vecGeraden,
        * vecPunkte;

   DBG1("Ausgabe");

   /******************************************************************
    ** Falls der Schalter bDruckeHtdos nicht gesetzt ist, werden HTDOs
    ** nicht gedruckt und die Funktion gleich wieder verlassen.
    ******************************************************************/
   if (!bIsTdo && (iMaxStufe != phtdo->iStufe) && !bDruckeHtdos ) {
      return;
   } /* endif */

   if (bHtdosProtokollieren && iMaxStufe == phtdo->iStufe) {
      AusgabeTestfall(phtdo);
   } /* endif */


   fprintf(OUTDAT, "\n Testfall ");
   for ( i=1; i <= phtdo->iStufe; ++i) {
      fprintf(OUTDAT, "%d.", vecTestfall[i]);
   } /* endfor */

   if (bIsTdo) {
      fprintf(OUTDAT, "\n TDO auf Stufe %d \n",phtdo->iStufe);
      ++lLoesungsAnzahl;
   } else {
      fprintf(OUTDAT, "\n Halb-TDO auf Stufe %d \n",phtdo->iStufe);
   } /* endif */


   if (0 == phtdo->iStufe % 2) {
      iGeradenAnzahl   = phtdo->iAltTypAnzahl;
      iPunkteAnzahl    = phtdo->iNeuTypAnzahl;
      piGeradenZaehler = &iAltTypZaehler;
      piPunkteZaehler  = &iNeuTypZaehler;
      vecGeraden       = phtdo->vecAlt;
      vecPunkte        = phtdo->vecNeu;
   } else {
      iGeradenAnzahl   = phtdo->iNeuTypAnzahl;
      iPunkteAnzahl    = phtdo->iAltTypAnzahl;
      piGeradenZaehler = &iNeuTypZaehler;
      piPunkteZaehler  = &iAltTypZaehler;
      vecGeraden       = phtdo->vecNeu;
      vecPunkte        = phtdo->vecAlt;
   }; /* endif */


   /******************************************************************
    ** Falls 2-Geraden gedruckt werden sollen, sind diese zunchst
    ** zu berechnen.
    ******************************************************************/
   if (bDruckeZweier) {
      iGeradenAnzahl += iBerechneZweier(phtdo,bIsTdo);
   } /* endif */

   fprintf(OUTDAT," Es gibt %d Geradentypen und %d Punkttypen.\n",
           iGeradenAnzahl, iPunkteAnzahl);
   fprintf(OUTDAT, "      ");

   for ( i = 0; i < iGeradenAnzahl ; ++i) {
      fprintf(OUTDAT, " %3d", vecGeraden[i]);
   } /* endfor */
   putc('\n', OUTDAT);

   for (*piPunkteZaehler = 0 ; *piPunkteZaehler < iPunkteAnzahl ; ++*piPunkteZaehler) {
      fprintf(OUTDAT, "%5d:", vecPunkte[*piPunkteZaehler]);
      for (*piGeradenZaehler = 0 ; *piGeradenZaehler < iGeradenAnzahl; ++*piGeradenZaehler) {
         fprintf(OUTDAT, " %3d", phtdo->vecNeu[iNeuTypZaehler]
                            * phtdo->matHtdo[iNeuTypZaehler][iAltTypZaehler]);
      } /* endfor */
      putc('\n', OUTDAT);
   } /* endfor */
   putc('\n', OUTDAT);

   if (bDruckeZweier) {
      if (1 == phtdo->iStufe % 2) {
         memset(phtdo->matHtdo[phtdo->iNeuTypAnzahl],0,
                sizeof(VECTOR)*(iGeradenAnzahl - phtdo->iNeuTypAnzahl));
      } else {
      } /* endif */
   } /* endif */

   DBG_F(fflush(OUTDAT));

} /* Ausgabe */





/******************************************************************
 ******************************************************************
 ** DebugHtdo
 ******************************************************************
 ******************************************************************/
static void DebugHtdo(HTDO *phtdo)
{
   INT  i,j;



   fprintf(OUTDAT, " vecTestfall:");
   for ( i=0; i < phtdo->iStufe; ++i) {
      fprintf(OUTDAT, "%d.", vecTestfall[i+1]);
   } /* endfor */

   fprintf(OUTDAT, "\n");

   fprintf(OUTDAT, " iStufe = %d \n", phtdo->iStufe );

   fprintf(OUTDAT, " iNeuAnzahl = %d, iAltAnzahl = %d \n",
                      phtdo->iNeuAnzahl, phtdo->iAltAnzahl );

   fprintf(OUTDAT, " iNeuTypAnzahlVorher = %d, iAltTypAnzahlVorher = %d \n",
                      phtdo->iNeuTypAnzahlVorher, phtdo->iAltTypAnzahlVorher );

   fprintf(OUTDAT, " iNeuTypAnzahl = %d, iAltTypAnzahl = %d \n",
                      phtdo->iNeuTypAnzahl, phtdo->iAltTypAnzahl );

   fprintf(OUTDAT, " vecNeuVorher:");
   for ( i=0; i < phtdo->iNeuTypAnzahlVorher; ++i) {
      fprintf(OUTDAT, "%3d", phtdo->vecNeuVorher[i]);
   } /* endfor */
   fprintf(OUTDAT, "\n");

   fprintf(OUTDAT, " vecAltVorher:");
   for ( i=0; i < phtdo->iAltTypAnzahlVorher; ++i) {
      fprintf(OUTDAT, "%3d", phtdo->vecAltVorher[i]);
   } /* endfor */
   fprintf(OUTDAT, "\n");

   fprintf(OUTDAT, "Letzte Exemplare der NeuTypenVorher: \n");
   for ( i=0; i < phtdo->iNeuTypAnzahlVorher ; ++i) {
      fprintf(OUTDAT, "%3d", phtdo->vecNeuTypVorherEnde[i]);
   } /* endfor */
   fprintf(OUTDAT, "\n");

   fprintf(OUTDAT, "Letzte AltTypen der AltTypenVorher: \n");
   for ( i=0; i < phtdo->iAltTypAnzahlVorher ; ++i) {
      fprintf(OUTDAT, "%3d", phtdo->vecAltTypVorherEnde[i]);
   } /* endfor */
   fprintf(OUTDAT, "\n");

   fprintf(OUTDAT, " vecNeu:");
   for ( i=0; i < phtdo->iNeuTypAnzahl; ++i) {
      fprintf(OUTDAT, "%3d", phtdo->vecNeu[i]);
   } /* endfor */
   fprintf(OUTDAT, "\n");

   fprintf(OUTDAT, " vecAlt:");
   for ( i=0; i < phtdo->iAltTypAnzahl; ++i) {
      fprintf(OUTDAT, "%3d", phtdo->vecAlt[i]);
   } /* endfor */
   fprintf(OUTDAT, "\n");

   fprintf(OUTDAT, "Matrix der Inzidenzen: \n");
   for ( i=0; i < phtdo->iNeuAnzahl ; ++i) {
      for ( j=0; j < phtdo->iAltTypAnzahl; ++j) {
         fprintf(OUTDAT, "%3d", phtdo->matInz[i][j]);
      } /* endfor */
      fprintf(OUTDAT, "\n");
   } /* endfor */


} /* DebugHtdo */



/******************************************************************
 ******************************************************************
 ** AusgabeTestfall
 ******************************************************************
 ******************************************************************/
static void AusgabeTestfall(HTDO *phtdo)
{
   INT  i,j;



   fprintf(TESTFALLOUT, "Testfall:");
   for ( i=0; i < phtdo->iStufe; ++i) {
      fprintf(TESTFALLOUT, "%d.", vecTestfall[i+1]);
   } /* endfor */
   fprintf(TESTFALLOUT, "\n");

   fprintf(TESTFALLOUT, " iNeuAnzahl = %d, iAltAnzahl = %d \n",
                      phtdo->iNeuAnzahl, phtdo->iAltAnzahl );

   fprintf(TESTFALLOUT, " iNeuTypAnzahlVorher = %d, iAltTypAnzahlVorher = %d \n",
                      phtdo->iNeuTypAnzahlVorher, phtdo->iAltTypAnzahlVorher );

   fprintf(TESTFALLOUT, " iNeuTypAnzahl = %d, iAltTypAnzahl = %d \n",
                      phtdo->iNeuTypAnzahl, phtdo->iAltTypAnzahl );

   fprintf(TESTFALLOUT, " vecNeuVorher:");
   for ( i=0; i < phtdo->iNeuTypAnzahlVorher; ++i) {
      fprintf(TESTFALLOUT, "%3d", phtdo->vecNeuVorher[i]);
   } /* endfor */
   fprintf(TESTFALLOUT, "\n");

   fprintf(TESTFALLOUT, " vecAltVorher:");
   for ( i=0; i < phtdo->iAltTypAnzahlVorher; ++i) {
      fprintf(TESTFALLOUT, "%3d", phtdo->vecAltVorher[i]);
   } /* endfor */
   fprintf(TESTFALLOUT, "\n");

   fprintf(TESTFALLOUT, "Letzte Exemplare der NeuTypenVorher: \n");
   for ( i=0; i < phtdo->iNeuTypAnzahlVorher ; ++i) {
      fprintf(TESTFALLOUT, "%3d", phtdo->vecNeuTypVorherEnde[i]);
   } /* endfor */
   fprintf(TESTFALLOUT, "\n");

   fprintf(TESTFALLOUT, "Letzte AltTypen der AltTypenVorher: \n");
   for ( i=0; i < phtdo->iAltTypAnzahlVorher ; ++i) {
      fprintf(TESTFALLOUT, "%3d", phtdo->vecAltTypVorherEnde[i]);
   } /* endfor */
   fprintf(TESTFALLOUT, "\n");

   fprintf(TESTFALLOUT, " vecNeu:");
   for ( i=0; i < phtdo->iNeuTypAnzahl; ++i) {
      fprintf(TESTFALLOUT, "%3d", phtdo->vecNeu[i]);
   } /* endfor */
   fprintf(TESTFALLOUT, "\n");

   fprintf(TESTFALLOUT, " vecAlt:");
   for ( i=0; i < phtdo->iAltTypAnzahl; ++i) {
      fprintf(TESTFALLOUT, "%3d", phtdo->vecAlt[i]);
   } /* endfor */
   fprintf(TESTFALLOUT, "\n");

   fprintf(TESTFALLOUT, "Matrix der Inzidenzen: \n");
   for ( i=0; i < phtdo->iNeuTypAnzahl ; ++i) {
      for ( j=0; j < phtdo->iAltTypAnzahl; ++j) {
         fprintf(TESTFALLOUT, "%3d", phtdo->matHtdo[i][j]);
      } /* endfor */
      fprintf(TESTFALLOUT, "\n");
   } /* endfor */

} /* AusgabeTestfall */



/******************************************************************
 ******************************************************************
 ** SuchenStartTestfall
 ******************************************************************
 ******************************************************************/
static BOOL bSuchenStartTestfall(HTDO *phtdo,char sPuffer[80])
{
   char *pch;

   /******************************************************************
    ** Suchen des Testfalls
    ******************************************************************/
   do {
     pch = fgets(sPuffer,80 /*sizeof(sPuffer)*/, TESTFALLIN);
   } while ( pch && memcmp(sPuffer+9,sStartTestfall,lLenStartTestfall) ); /* enddo */
   DBG2("%s\n",sPuffer);
   if (NULL == pch) {
      return FALSE;
   } else {
      return TRUE;
   } /* endif */

} /* SuchenStartTestfall */



/******************************************************************
 ******************************************************************
 ** SuchenNextTestfall
 ******************************************************************
 ******************************************************************/
static BOOL bSuchenNextTestfall(HTDO *phtdo,char sPuffer[80])
{
   char *pch;

   /******************************************************************
    ** Suchen des Testfalls
    ******************************************************************/
   do {
     pch = fgets(sPuffer,80 /*sizeof(sPuffer)*/, TESTFALLIN);
   } while ( pch && memcmp(sPuffer,"Testfall:",9)); /* enddo */
   DBG2("%s\n",sPuffer);
   if (NULL == pch) {
      return FALSE;
   } else {
      return TRUE;
   } /* endif */

} /* SuchenNextTestfall */





/******************************************************************
 ******************************************************************
 ** EinlesenTestfall
 ******************************************************************
 ******************************************************************/
static BOOL bEinlesenTestfall(HTDO *phtdo,int iNummer)
{
   char sPuffer[80];
   char *pch;
   INT  i,j;
   BOOL b;

   if (1 == iNummer) {
      b = bSuchenStartTestfall(phtdo,sPuffer);
   } else {
      b = bSuchenNextTestfall(phtdo,sPuffer);
   } /* endif */
   if (!b) {
      return FALSE;
   } /* endif */

   /******************************************************************
    ** Ueberschreiben des fuehrenden Wortes "Testfall:"
    ******************************************************************/
   memcpy(sPuffer, sPuffer+9, sizeof(sPuffer)-9);

   /******************************************************************
    ** Einlesen der Testfall-Nummer
    ******************************************************************/
   for ( i=0; '0'<=sPuffer[0] && '9'>=sPuffer[0]; ++i) {
      for ( j = 0; sPuffer[j]!='.' && j<sizeof(sPuffer) ; ++j) {
         ;
      } /* endfor */
      sPuffer[j] = ' ';
      sscanf(sPuffer,"%d", &aiStartTestfall[i+1]);
      vecTestfall[i+1] = aiStartTestfall[i+1];
      memcpy(sPuffer, sPuffer+j+1, sizeof(sPuffer)-j-1);
   }; /* endfor */
   DBG8("Start-Testfall: %d.%d.%d.%d.%d.%d.%d. \n",
        aiStartTestfall[1],aiStartTestfall[2],aiStartTestfall[3],aiStartTestfall[4],
        aiStartTestfall[5],aiStartTestfall[6],aiStartTestfall[7]);

   phtdo->iStufe = i;


   /******************************************************************
    ** Einlesen der restlichen Daten
    ******************************************************************/
   fscanf(TESTFALLIN, " iNeuAnzahl = %d, iAltAnzahl = %d \n",
                      &phtdo->iNeuAnzahl, &phtdo->iAltAnzahl );

   fscanf(TESTFALLIN, " iNeuTypAnzahlVorher = %d, iAltTypAnzahlVorher = %d \n",
                      &phtdo->iNeuTypAnzahlVorher, &phtdo->iAltTypAnzahlVorher );

   fscanf(TESTFALLIN, " iNeuTypAnzahl = %d, iAltTypAnzahl = %d \n",
                      &phtdo->iNeuTypAnzahl, &phtdo->iAltTypAnzahl );

   fscanf(TESTFALLIN, " vecNeuVorher:");
   for ( i=0; i < phtdo->iNeuTypAnzahlVorher; ++i) {
      fscanf(TESTFALLIN, "%3d", &phtdo->vecNeuVorher[i]);
   } /* endfor */
   fscanf(TESTFALLIN, "\n");

   fscanf(TESTFALLIN, " vecAltVorher:");
   for ( i=0; i < phtdo->iAltTypAnzahlVorher; ++i) {
      fscanf(TESTFALLIN, "%3d", &phtdo->vecAltVorher[i]);
   } /* endfor */
   fscanf(TESTFALLIN, "\n");

   fscanf(TESTFALLIN, "Letzte Exemplare der NeuTypenVorher: \n");
   for ( i=0; i < phtdo->iNeuTypAnzahlVorher ; ++i) {
      fscanf(TESTFALLIN, "%3d", &phtdo->vecNeuTypVorherEnde[i]);
   } /* endfor */
   fscanf(TESTFALLIN, "\n");

   fscanf(TESTFALLIN, "Letzte AltTypen der AltTypenVorher: \n");
   for ( i=0; i < phtdo->iAltTypAnzahlVorher ; ++i) {
      fscanf(TESTFALLIN, "%3d", &phtdo->vecAltTypVorherEnde[i]);
   } /* endfor */
   fscanf(TESTFALLIN, "\n");

   fscanf(TESTFALLIN, " vecNeu:");
   for ( i=0; i < phtdo->iNeuTypAnzahl; ++i) {
      fscanf(TESTFALLIN, "%d", &phtdo->vecNeu[i]);
   } /* endfor */
   fscanf(TESTFALLIN, "\n");

   fscanf(TESTFALLIN, " vecAlt:");
   for ( i=0; i < phtdo->iAltTypAnzahl; ++i) {
      fscanf(TESTFALLIN, "%d", &phtdo->vecAlt[i]);
   } /* endfor */
   fscanf(TESTFALLIN, "\n");

   fscanf(TESTFALLIN, "Matrix der Inzidenzen: \n");
   for ( i=0; i < phtdo->iNeuTypAnzahl ; ++i) {
      for ( j=0; j < phtdo->iAltTypAnzahl; ++j) {
         fscanf(TESTFALLIN, "%d", &phtdo->matHtdo[i][j]);
      } /* endfor */
      fscanf(TESTFALLIN, "\n");
   } /* endfor */

   DBG_F(DebugHtdo(phtdo));
   Ausgabe(phtdo,FALSE);
   fflush(OUTDAT);

   return TRUE;
} /* bEinlesenTestfall */





/******************************************************************
 ******************************************************************
 ** AusgabeParms
 ******************************************************************
 ******************************************************************/
static void AusgabeParms(HTDO_PARMS *pparms,HTDO *phtdo)
{
   INT i,j;

   fprintf(OUTDAT, "\n AUSGABE PARMS\n");

   fprintf(OUTDAT, "NeuExemplar:%d NeuTypVorher:%d NeuTypVorherIndex: %d\n",
           pparms->iNeuExemplar,pparms->iNeuTypVorher, pparms->iNeuTypVorherIndex);
   fprintf(OUTDAT, "     AltTyp:%d AltTypVorher:%d \n",
           pparms->iAltTyp,pparms->iAltTypVorher);

   fprintf(OUTDAT, "Ordnen:%d Minimum:%d \n",
           pparms->bOrdnen, pparms->iMinimum);

   fprintf(OUTDAT, "NeuTypVorherIndices: \n");
   for ( i=0; i < phtdo->iNeuTypAnzahlVorher ; ++i) {
      fprintf(OUTDAT, "%3d", pparms->vecNeuTypVorherIndex[i]);
   } /* endfor */
   fprintf(OUTDAT, "\n");

   fprintf(OUTDAT, "Matrix der Inzidenzen zwischen den Alttypen: \n");
   for ( i=0; i < phtdo->iAltTypAnzahl ; ++i) {
      for ( j=0; j < phtdo->iAltTypAnzahl; ++j) {
         fprintf(OUTDAT, "%3d", pparms->matAltTypInzidenzen[i][j]);
      } /* endfor */
      fprintf(OUTDAT, "\n");
   } /* endfor */

   fprintf(OUTDAT, "Matrix der AltTypFahnen zu den NeuTypenVorher:\n");
   for ( i=0; i < phtdo->iNeuTypAnzahlVorher ; ++i) {
      for ( j=0; j < phtdo->iAltTypAnzahl; ++j) {
         fprintf(OUTDAT, "%3d", pparms->matAltTypFahnen[i][j]);
      } /* endfor */
      fprintf(OUTDAT, "\n");
   } /* endfor */

   fprintf(OUTDAT, "Matrix der NeuTypVorherInzidenzen fuer die NeuExemplare:\n");
   for ( i=0; i < phtdo->iNeuAnzahl ; ++i) {
      for ( j=0; j < phtdo->iNeuTypAnzahlVorher; ++j) {
         fprintf(OUTDAT, "%3d", pparms->matNeuTypInzidenzen[i][j]);
      } /* endfor */
      fprintf(OUTDAT, "\n");
   } /* endfor */

   fprintf(OUTDAT, "Matrix der NeuTypVorherFahnen fuer die NeuExemplare:\n");
   for ( i=0; i < phtdo->iNeuAnzahl ; ++i) {
      for ( j=0; j < phtdo->iAltTypAnzahlVorher; ++j) {
         fprintf(OUTDAT, "%3d", pparms->matNeuTypFahnenVorher[i][j]);
      } /* endfor */
      fprintf(OUTDAT, "\n");
   } /* endfor */

} /* AusgabeParms */




/******************************************************************
 ******************************************************************
 ** Ausgabe_Parms
 ******************************************************************
 ******************************************************************/
static void Ausgabe_Parms(HTDO1_PARMS *pparms,HTDO *phtdo)
{
   INT i,j;

   fprintf(OUTDAT, "\n AUSGABE HTDO1-PARMS\n");


   fprintf(OUTDAT, " iMinimum = %d, iZweier = %d \n",
                      pparms->iMinimum, pparms->iZweier );
   fflush(OUTDAT);

   fprintf(OUTDAT, " vecFahnen:");
   for ( i=0; i < phtdo->iAltTypAnzahl; ++i) {
      fprintf(OUTDAT, "%3d", pparms->vecFahnen[i]);
   } /* endfor */
   fprintf(OUTDAT, "\n");

   fprintf(OUTDAT, " vecPktVerbindungen:");
   for ( i=0; i < phtdo->iNeuAnzahl; ++i) {
      fprintf(OUTDAT, "%3d", pparms->vecPktVerbindungen[i]);
   } /* endfor */
   fprintf(OUTDAT, "\n");
   fflush(OUTDAT);

   fprintf(OUTDAT, " matGerSchneidungen:\n");
   for ( i=0; i < phtdo->iAltTypAnzahl; ++i) {
      for ( j=0; j < phtdo->iAltTypAnzahl; ++j) {
         fprintf(OUTDAT, "%3d", pparms->matGerSchneidungen[i][j]);
      } /* endfor */
      fprintf(OUTDAT, "\n");
   } /* endfor */
   fprintf(OUTDAT, "\n");

   fflush(OUTDAT);
} /* Ausgabe_Parms */



/******************************************************************
 ******************************************************************
 ** Debug_
 ******************************************************************
 ******************************************************************/
static void Debug_(INT i, INT j,HTDO *phtdo,HTDO1_PARMS *pparms)
{
   DBG3(" i = %d, j= %d \n", i,j);
   DebugHtdo(phtdo);
   Ausgabe_Parms(pparms, phtdo);

} /* Debug_ */


/***************************************************************************
 ***************************************************************************
 ** bNormalisierteHtdoIstTdo
 ** Gleiche Exemplare in matInz werden in matHtdo zusammengefasst
 ** TRUE  = die matHtdo ist eine TDO
 ** FALSE = sie ist keine TDO
 ***************************************************************************
 ***************************************************************************/
static BOOL bNormalisierteHtdoIstTdo(HTDO* phtdo)
{
   LONG lHtdo = 0,
        lInz  = 0,
        lExem = 0;


   DBG1("bNormalisierteHtdoIstTdo\n");
   /******************************************************************
    ** Zunaechst wird die Normalisierung durchgefhrt
    ******************************************************************/
   memcpy(phtdo->matHtdo[lHtdo], phtdo->matInz[lInz], sizeof(VECTOR));
   DBG4("%ld: %d %d \n",lHtdo,phtdo->matHtdo[lHtdo][0],phtdo->matHtdo[lHtdo][1]);
   lExem = 1;

   for (lInz = 1; lInz < phtdo->iNeuAnzahl; ++lInz ) {
      /******************************************************************
       ** Wenn sich der Vector in der matInz nicht verndert hat, wird
       ** nur der ExemplarZaehler erhoeht.
       ** Sonst muss der neue Typ in die matHtdo uebertragen werden und der
       ** ExemplarZaehler wird auf 1 zurueckgesetzt.
       ******************************************************************/
      if (0 == memcmp(phtdo->matHtdo[lHtdo], phtdo->matInz[lInz], sizeof(VECTOR))) {
         ++lExem;
      } else {
         phtdo->vecNeu[lHtdo] = lExem;
         ++lHtdo;
         memcpy(phtdo->matHtdo[lHtdo], phtdo->matInz[lInz], sizeof(VECTOR));
         DBG4("%ld: %d %d \n",lHtdo,phtdo->matHtdo[lHtdo][0],phtdo->matHtdo[lHtdo][1]);
         lExem = 1;
      } /* endif */
   } /* endfor */

   phtdo->vecNeu[lHtdo] = lExem;
   phtdo->iNeuTypAnzahl = lHtdo +1;

   /******************************************************************
    ** Die Normalisierung ist beendet, jetzt wird getestet, ob bereits
    ** eine TDO erreicht ist. Dazu muss nur berprft werden,
    ** ob die neue Htdo zu einer weiteren Spezifizierung der NeuKlasse
    ** gefhrt hat.
    ******************************************************************/
   return (phtdo->iNeuTypAnzahl == phtdo->iNeuTypAnzahlVorher);

} /* bNormalisierteHtdoIstTdo */




/***************************************************************************
 ***************************************************************************
 ** AusgabeMaxfit
 ***************************************************************************
 ***************************************************************************/
static void AusgabeMaxfit()
{
   INT i,j;

   for ( i=1; i<=10; ++i) {
      for ( j=1 ; j<=10; ++j) {
         fprintf(OUTDAT, "%2d ",matMaxfit[i][j]);
      } /* endfor */
      fprintf(OUTDAT, "\n");
   } /* endfor */
} /* AusgabeMaxfit */



/***************************************************************************
 ***************************************************************************
 ** InitMaxfit
 ** Diese Funktion berechnet die maximale Anzahl von Inzidenzen, die in einen
 ** (i,j)-Block passen und trgt dies in die Maxfit-Matrix ein.
 ***************************************************************************
 ***************************************************************************/
static void InitMaxfit()
{
   LONG lMax,
        lMin,
        lInz,
        lGki,
        lVerb;

   DBG1("InitMaxfit \n");

   /***************************************************************************
    * lMax ist der groessere der Indices im (i,j)-Block.
    * Fuer jede Anzahl von Inzidenzen lInz auf diesem Block wird die
    * Mindestanzahl von Verbindungen lVerb berechnet, die auf den
    * lMin Punkten gezogen werden.
    * Solange (lMin ueber 2), die Anzahl der moeglichen Verbindungen
    * auf lMin Punkten, groessergleich lVerb ist, ist lInz als
    * Anzahl von Inzidenzen im (i,j)-Block zugelassen.
    ***************************************************************************/
   for (lMax = 1; lMax < MAX_VARANZ -1 ; ++lMax) {
      for (lInz = lMax , lMin = 1; lMax >= lMin; ++lInz) {
         lGki  = lInz / lMax;
         lVerb = lMax * lGki * (lGki-1) / 2 + (lInz % lMax) * lGki;
         if (lMin * (lMin-1) /2 >= lVerb) {
            matMaxfit[lMin][lMax] = lInz;
            matMaxfit[lMax][lMin] = lInz;
         } else {
            ++lMin;
            matMaxfit[lMin][lMax] = lInz;
            matMaxfit[lMax][lMin] = lInz;
         } /* endif */
 /*       for (lMin = lMax; (lMin>0) && (lMin * (lMin-1) /2 >= lVerb) ; --lMin) {
            matMaxfit[lMin][lMax] = lInz;
            matMaxfit[lMax][lMin] = lInz;
         }  endfor */
      } /* endfor */
   } /* endfor */

   DBG_F(AusgabeMaxfit());
} /* InitMaxfit */


/***************************************************************************
 ***************************************************************************
 ** bPruefeMaxfit
 ** Diese Funktion erkennt anhand geometrischer Kriterien, wenn die HTDO
 ** nicht zu einer TDO ausgebaut werden kann (maxfit-Formel).
 ** TRUE  = die matHtdo kann zu einer TDO werden
 ** FALSE = es ist keine TDO moeglich
 ***************************************************************************
 ***************************************************************************/
static BOOL bPruefeMaxfit(HTDO* phtdo)
{
   INT i,j,k;

   DBG1("PruefeMaxfit ");

   /******************************************************************
    ** Zunaechst der einstufige Test (anhand der HTDO-Blocks)
    ******************************************************************/
   for ( i=0; i<phtdo->iNeuTypAnzahl; ++i) {
      for ( j=0; j<phtdo->iAltTypAnzahl; ++j ) {
         if (phtdo->matHtdo[i][j] * phtdo->vecNeu[i]
             > matMaxfit[phtdo->vecNeu[i]][phtdo->vecAlt[j]]) {
            DBG2_5("i=%d, j=%d, vecNeu[i]=%d Maxfit=%d Fehler1\n",
                       i,j,phtdo->vecNeu[i],
                       matMaxfit[phtdo->vecNeu[i]][phtdo->vecAlt[j]] );
            DBG_F(Ausgabe(phtdo,FALSE));
            return FALSE;
         } /* endif */
      } /* endfor */
   } /* endfor */

   return TRUE;
}



/***************************************************************************
 ***************************************************************************
 ** bPruefeMaxfit2
 ** Diese Funktion erkennt anhand geometrischer Kriterien, wenn die HTDO
 ** nicht zu einer TDO ausgebaut werden kann (maxfit-Formel).
 ** Dies ist die erweiterte Version der Funktion bPruefeMaxfit.
 ** Hier werden die gekoppelten HTDO-Blocks gegen die Maxfit-Tabelle
 ** geprueft.
 ** TRUE  = die matHtdo kann zu einer TDO werden
 ** FALSE = es ist keine TDO moeglich
 ***************************************************************************
 ***************************************************************************/
static BOOL bPruefeMaxfit2(HTDO* phtdo)
{
   INT i,j,k;

   DBG1("PruefeMaxfit2 ");

   /******************************************************************
    ** Nun der zweistufige Test (anhand von gekoppelten HTDO-Blocks)
    ******************************************************************/
   for ( i=0; i<phtdo->iNeuTypAnzahl -1; ++i) {
      for ( k=i+1; k<phtdo->iNeuTypAnzahl; ++k) {
         for ( j=0; j<phtdo->iAltTypAnzahl; ++j ) {
            DBG3("%d <= %d \n",
                phtdo->matHtdo[i][j] * phtdo->vecNeu[i]
                + phtdo->matHtdo[k][j] * phtdo->vecNeu[k],
                matMaxfit[phtdo->vecNeu[i]+phtdo->vecNeu[k]][phtdo->vecAlt[j]]);
            if (phtdo->matHtdo[i][j] * phtdo->vecNeu[i]
                + phtdo->matHtdo[k][j] * phtdo->vecNeu[k]
                > matMaxfit[phtdo->vecNeu[i]+phtdo->vecNeu[k]][phtdo->vecAlt[j]]) {
               DBG1(" Fehler2\n");
               DBG_F(Ausgabe(phtdo,FALSE));
               return FALSE;
            } /* endif */
         } /* endfor */
      } /* endfor */
   } /* endfor */

   /******************************************************************
    ** Nun der zweistufige Test (anhand von gekoppelten HTDO-Blocks)
    ******************************************************************/
   for ( i=0; i<phtdo->iNeuTypAnzahl; ++i) {
      for ( j=0; j<phtdo->iAltTypAnzahl -1; ++j) {
         for ( k=j+1; k<phtdo->iAltTypAnzahl; ++k ) {
            DBG3("%d <= %d \n",
                phtdo->matHtdo[i][j] * phtdo->vecNeu[i]
                + phtdo->matHtdo[i][k] * phtdo->vecNeu[i],
                matMaxfit[phtdo->vecNeu[i]][phtdo->vecAlt[j]+phtdo->vecAlt[k]]);
            if (phtdo->matHtdo[i][j] * phtdo->vecNeu[i]
                + phtdo->matHtdo[i][k] * phtdo->vecNeu[i]
                > matMaxfit[phtdo->vecNeu[i]][phtdo->vecAlt[j]+phtdo->vecAlt[k]]) {
               DBG1(" Fehler3\n");
               DBG_F(Ausgabe(phtdo,FALSE));
               return FALSE;
            } /* endif */
         } /* endfor */
      } /* endfor */
   } /* endfor */

DBG1(" TRUE\n");
return TRUE;
} /* bPruefeMaxfit2 */




/***************************************************************************
 ***************************************************************************
 ** bPruefeInzidenzen
 ** Diese Funktion erkennt anhand geometrischer Kriterien, wenn die HTDO
 ** nicht zu einer TDO ausgebaut werden kann.
 ** Nach der Normalisierung darf kein Block entstanden sein, der
 ** zu viele innere Verbindungen aufweist.
 ** TRUE  = die matHtdo kann zu einer TDO werden
 ** FALSE = es ist keine TDO moeglich
 ***************************************************************************
 ***************************************************************************/
static BOOL bPruefeInzidenzen(HTDO* phtdo)
{
   INT i,i2,j,iHilf,iSum,iMaxBlock, iAnzBlock;

   DBG1("PruefeInzidenzen ");

   /******************************************************************
    ** iSum zhlt die Inzidenzen auf dem i-ten Neutyp
    ******************************************************************/
   for ( i=0; i<phtdo->iNeuTypAnzahl; ++i) {
      iSum = 0;
      for ( j=0; j<phtdo->iAltTypAnzahl; ++j ) {
         /* Groesster Unterblock */
         iMaxBlock = phtdo->matHtdo[i][j] * phtdo->vecNeu[i] / phtdo->vecAlt[j] +1;
         if (iMaxBlock > 1) {
            /* Anzahl davon */
            iAnzBlock = phtdo->matHtdo[i][j] * phtdo->vecNeu[i] % phtdo->vecAlt[j];
            iSum += iMaxBlock * (iMaxBlock -1) * iAnzBlock;
            iSum += (iMaxBlock-1) * (iMaxBlock -2) * (phtdo->vecAlt[j] - iAnzBlock);
         } /* endif */
      } /* endfor */
      if (iSum > phtdo->vecNeu[i]*(phtdo->vecNeu[i] -1)) {
         return FALSE;
      } /* endif */
   } /* endfor */

   /******************************************************************
    ** iSum zhlt die Inzidenzen auf der Vereinigung des
    ** i-ten und i2-ten Neutyp
    ******************************************************************/
   for ( i=0; i<phtdo->iNeuTypAnzahl-1; ++i) {
      for ( i2=i+1; i2<phtdo->iNeuTypAnzahl; ++i2) {
         iSum = 0;
         for ( j=0; j<phtdo->iAltTypAnzahl; ++j ) {
            /* Groesster Unterblock */
            iHilf     = (phtdo->matHtdo[i][j] * phtdo->vecNeu[i])+(phtdo->matHtdo[i2][j] * phtdo->vecNeu[i2]);
            iMaxBlock = iHilf / phtdo->vecAlt[j] +1;
            if (iMaxBlock > 1) {
               /* Anzahl davon */
               iAnzBlock = iHilf % phtdo->vecAlt[j];
               iSum += iMaxBlock * (iMaxBlock -1) * iAnzBlock;
               iSum += (iMaxBlock-1) * (iMaxBlock -2) * (phtdo->vecAlt[j] - iAnzBlock);
            } /* endif */
         } /* endfor */
         if (iSum > (phtdo->vecNeu[i]+phtdo->vecNeu[i2])*(phtdo->vecNeu[i]+phtdo->vecNeu[i2] -1)) {
            return FALSE;
         } /* endif */
      } /* endfor */
   } /* endfor */

   return TRUE;
} /* bPruefeInzidenzen */



/***************************************************************************
 ***************************************************************************
 ** bTdoPlausi
 ** Diese Funktion erkennt anhand geometrischer Kriterien, wenn die HTDO
 ** nicht zu einer TDO ausgebaut werden kann. Diese Funktion macht nur
 ** bei der ersten Punkttypen-Verteilung (also bei Stufe 2) Sinn.
 ** TRUE  = die matHtdo kann zu einer TDO werden
 ** FALSE = es ist keine TDO moeglich
 ***************************************************************************
 ***************************************************************************/
static BOOL bTdoPlausi(HTDO* phtdo, GERADENFALL* pgf)
{
   INT i,j,iIndMaxGer,iAnzPktInzGer;

   DBG1("bTdoPlausi ");

   /******************************************************************
    ** iIndMaxGer ist der Index des Punkttyps mit der hchsten Anzahl
    ** an Geraden des i-Typs
    ** iAnzPktInzGer ist die Gesamtanzahl an Punkten, die mit i-Typ-Geraden
    ** inzidieren
    ******************************************************************/
   for ( i=0; i<phtdo->iAltTypAnzahl; ++i) {
      iIndMaxGer    = 0;
      iAnzPktInzGer = 0;

      if ( phtdo->matHtdo[0][i] > 0 ) {
         iAnzPktInzGer = phtdo->vecNeu[0];
      } /* endif */

      for ( j=1; j<phtdo->iNeuTypAnzahl; ++j ) {
         if ( phtdo->matHtdo[j][i] > phtdo->matHtdo[iIndMaxGer][i] ) {
            iIndMaxGer = j;
         } /* endif */
         if ( phtdo->matHtdo[j][i] > 0 ) {
            iAnzPktInzGer += phtdo->vecNeu[j];
         } /* endif */
      } /* endfor */

      /******************************************************************
       ** Ein Punkt des Typs iIndMaxGer darf ber i-Typ-Geraden nur mit hchstens
       ** (iAnzPktInzGer -1) Punkten inzidieren.
       ******************************************************************/
      DBG2_4("IndMax %d, MinGer %d, AnzGer %d \n",
             iIndMaxGer, phtdo->matHtdo[iIndMaxGer][i] * (pgf->agt[i].iLaenge -1),iAnzPktInzGer-1);
      if (phtdo->matHtdo[iIndMaxGer][i] * (pgf->agt[i].iLaenge -1) > iAnzPktInzGer -1) {
         return FALSE;
      } /* endif */
   } /* endfor */

   return TRUE;
} /* bTdoPlausi */



/******************************************************************
 ******************************************************************
 ** InitialisiereParms
 ******************************************************************
 ******************************************************************/
static VOID InitialisiereParms(HTDO* phtdo, HTDO *phtdoAlt, HTDO_PARMS *pparms)
{
   INT i,j,i2,j2,i3;
   INT iSum;

   DBG2_("InitialisiereParms\n");

  /******************************************************************
   *   Initialisieren der Matrix der Inzidenzen der AltKlasse
   *   Sie enthlt die Anzahl der Inzidenzen, die zwischen den
   *   AltTypen i und i2 bestehen drfen (bzw. das doppelte dieser
   *   Anzahl bei i = i2 zur Vereinfachung der Rechnung)
   ******************************************************************/
  for ( j = 0; j < phtdo->iAltTypAnzahl; ++j) {
     pparms->matAltTypInzidenzen[j][j] = phtdo->vecAlt[j]
                                         * (phtdo->vecAlt[j] - 1);
     for ( j2 = j+1; j2 < phtdo->iAltTypAnzahl; ++j2) {
        pparms->matAltTypInzidenzen[j][j2] = phtdo->vecAlt[j]
                                             * phtdo->vecAlt[j2];
     } /* endfor */
  } /* endfor */
  DBG1("InitialisiereParms1\n");

  /******************************************************************
   *   Initialisieren der AltTypFahnen-Matrix
   *   Er enthlt die Anzahl der Inzidenzen der NeuTypenVorher mit den
   *   einzelnen Typen der AltKlasse
   ******************************************************************/
  for ( i = 0; i < phtdo->iNeuTypAnzahlVorher; ++i) {
     for ( j = 0; j < phtdo->iAltTypAnzahl; ++j) {
        pparms->matAltTypFahnen[i][j] = phtdo->vecAlt[j]
                                        * phtdoAlt->matHtdo[j][i];
     } /* endfor */
  } /* endfor */
  DBG1("InitialisiereParms2\n");


  /******************************************************************
   *   Initialisieren der Fahnen-Matrix
   *   Sie enthlt die Anzahl der Inzidenzen der NeuExemplare mit den
   *   AltTypAnzahlVorher Typen der AltKlasse vor der letzten Spezifikation
   ******************************************************************/
  for ( i = 0, i2 = 0; i < phtdo->iNeuTypAnzahlVorher; ++i, ++i2) {
     for ( j = 0, j2 = 0; j < phtdo->iAltTypAnzahlVorher; ++j) {
        iSum = 0;
        for ( ; j2 <= phtdo->vecAltTypVorherEnde[j]; ++j2) {
           iSum += phtdoAlt->matHtdo[j2][i] * phtdo->vecAlt[j2];
        } /* endfor */
        pparms->matNeuTypFahnenVorher[i2][j] = iSum / phtdo->vecNeuVorher[i];
     } /* endfor */
     for ( ; i2 < phtdo->vecNeuTypVorherEnde[i]; ++i2) {
        memcpy(pparms->matNeuTypFahnenVorher[i2 +1],
               pparms->matNeuTypFahnenVorher[i2], sizeof(VECTOR));
     } /* endfor */
  } /* endfor */
  DBG1("InitialisiereParms3\n");


  /******************************************************************
   *   Initialisieren der Matrix der NeuTypInzidenzen
   *   Sie enthlt zu jedem NeuExemplar die Anzahl der Inzidenzen
   *   zu den iNeuTypAnzahlVorher Typen der eigenen Klasse
   ******************************************************************/
  for ( i = 0, i2 = 0; i < phtdo->iNeuTypAnzahlVorher; ++i, ++i2) {
     for ( i3 = 0; i3 < phtdo->iNeuTypAnzahlVorher; ++i3) {
        if ( i == i3) {
           pparms->matNeuTypInzidenzen[i2][i3] = phtdo->vecNeuVorher[i3] - 1;
        } else {
           pparms->matNeuTypInzidenzen[i2][i3] = phtdo->vecNeuVorher[i3];
        } /* endif */
     } /* endfor */
     for ( i3 = 0; i3 < phtdo->vecNeuVorher[i] -1; ++i3, ++i2) {
        memcpy(pparms->matNeuTypInzidenzen[i2 +1], pparms->matNeuTypInzidenzen[i2],
               sizeof(VECTOR));
     } /* endfor */
  } /* endfor */
  DBG1("InitialisiereParms4\n");

  /******************************************************************
   * Initialisierung des Vectors vecNeuTypVorherIndex.
   * Er enthaelt die Indices der NeuTypenVorher, geordnet nach ihren
   * Anzahlen (beginnend mit dem Kleinsten). Die NeuTypenVorher werden in
   * dieser Reihenfolge abgearbeitet.
   ******************************************************************/
  pparms->vecNeuTypVorherIndex[0] = 0;
  if (phtdo->iStufe % 2 == 0) {
     for ( i=1; i<phtdo->iNeuTypAnzahlVorher; ++i) {
        for (j=i;
             (j>0) && (phtdo->vecNeuVorher[i]
               < phtdo->vecNeuVorher[pparms->vecNeuTypVorherIndex[j-1]]);
             --j ) {
           pparms->vecNeuTypVorherIndex[j] = pparms->vecNeuTypVorherIndex[j-1];
        } /* endfor */
        pparms->vecNeuTypVorherIndex[j] = i;
     } /* endfor */
  } else {
     for ( i=1; i<phtdo->iNeuTypAnzahlVorher; ++i) {
        for (j=i;(j>0)
                 && (phtdo->vecNeuVorher[i]<6)
                 && (phtdo->vecNeuVorher[i]
                    < phtdo->vecNeuVorher[pparms->vecNeuTypVorherIndex[j-1]]);
             --j ) {
           pparms->vecNeuTypVorherIndex[j] = pparms->vecNeuTypVorherIndex[j-1];
        } /* endfor */
        pparms->vecNeuTypVorherIndex[j] = i;
     } /* endfor */
     /* for ( i=1; i<phtdo->iNeuTypAnzahlVorher; ++i) {
        pparms->vecNeuTypVorherIndex[i] = i; */
     /* }  endfor */
  } /* endif */
  DBG1("InitialisiereParms5\n");


  /******************************************************************
   *   Initialisieren von bOrdnen und den Indices
   ******************************************************************/
  pparms->bOrdnen              = FALSE;

  pparms->iNeuTypVorherIndex = 0;
  pparms->iNeuTypVorher      = pparms->vecNeuTypVorherIndex[0];
  if (pparms->iNeuTypVorher  == 0) {
     pparms->iNeuExemplar    = 0;
  } else {
     pparms->iNeuExemplar    =
                 phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher -1] +1;
  } /* endif */

  pparms->iAltTyp            = 0;
  pparms->iAltTypVorher      = 0;

  /**********************************************************************
   *  Die Anzahl der AltTypFahnen darf mit
   *  steigendem *pi nicht zunehmen, da die Geometrien lexikografisch
   *  geordnet werden muessen. Daraus ergibt sich ein Mindestwert fuer
   *  matInz[*pi][*pj].
   *  Dieses Minimum ist
   *  (Anzahl der vebliebenen Fahnen) div (Anzahl der verbl. NeuTypenVorher)
   *  + (1 falls Fahnen mod NeuTypenVorher >= 1).
   *  Dies entspricht der folgenden Formel.
   **********************************************************************/
  pparms->iMinimum = (pparms->matAltTypFahnen[pparms->iNeuTypVorher][pparms->iAltTyp]
                      + phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher])
                      / (phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher]
                         +1);


  DBG_F(AusgabeParms(pparms,phtdo));
  ++lInitAnz;

  return;
} /* InitialisiereParms */






/******************************************************************
 ******************************************************************
 ** bMaximiereInzidenzen
 ** TRUE  = der maximale Wert von matInz(i,j) konnte ermittelt werden
 ** FALSE = jeder denkbare Wert fhrt zu Widersprchen
 ******************************************************************
 ******************************************************************/
static BOOL bMaximiereInzidenzen(HTDO       * phtdo, HTDO *phtdoAlt,
                                 HTDO_PARMS * pparms )

{
   BOOL  bAufgabeGeloest = TRUE;
   INT    i    = pparms->iNeuExemplar;
   INT    j    = pparms->iAltTyp;
   INT    k;
   INT * piInz = &phtdo->matInz[i][j];
   TM(clock_t cl  = clock();)


   DBG2("MaximiereInzidenzen %ld\n",clock()-cl);
   /******************************************************************
    * Die Anzahl der Inzidenzen wird initialisiert mit dem Minimum aus
    * der Anzahl der Fahnen auf dem AltTyp
    * und der Anzahl der Exemplare des AltTyps
    ******************************************************************/
   *piInz = MIN(pparms->matAltTypFahnen[pparms->iNeuTypVorher][j],
                phtdo->vecAlt[j]);

   /******************************************************************
    * Falls geordnet werden muss, begrenzt der (i-1)-Punkt die Inzidenzen
    ******************************************************************/
   if (pparms->bOrdnen) {
      *piInz = MIN(*piInz, phtdo->matInz[i-1][j]);
   } /* endif */

   /******************************************************************
    * Die Anzahl der Inzidenzen darf nicht groesser sein als die
    * Anzahl der Inzidenzen des NeuTypVorher mit dem AltTypVorher
    ******************************************************************/
   *piInz = MIN(*piInz, pparms->matNeuTypFahnenVorher[i][pparms->iAltTypVorher]);


   /******************************************************************
    * Die Schneidungen der AltTypen sind durch matAltTypInzi. begrenzt *
    ******************************************************************/
   for (k=0; (*piInz > 0) && (k < j) ; k++) {
      if (0 < phtdo->matInz[i][k]) {
         *piInz = MIN(*piInz,
                     (pparms->matAltTypInzidenzen[k][j] / phtdo->matInz[i][k])
                  );
      } /* endif */
   } /* endfor */
   while (*piInz * (*piInz -1) > pparms->matAltTypInzidenzen[j][j]) {
      --*piInz;
   } /* endwhile */


   /******************************************************************
    * Das NeuExemplar i wird (*piInz)-mal mit NeuTypen verbunden.
    * Die Anzahl der Verbindungen berechnet sich aus dem AltTyp j,
    * beschrieben in phtdoAlt->matHtdo.
    * Die Anzahl der neuen Verbindungen darf den Wert in der
    * matNeuTypInzidenzen nicht bersteigen.
    ******************************************************************/
   for ( k=0; (*piInz > 0) && (k < phtdo->iNeuTypAnzahlVorher) ; k++) {
      if (k == pparms->iNeuTypVorher) {
         if (0 < phtdoAlt->matHtdo[j][k] -1) {
            *piInz = MIN(*piInz,
                         (pparms->matNeuTypInzidenzen[i][k]
                         / (phtdoAlt->matHtdo[j][k] -1) )
                        );
         } /* endif */
      } else {
         if (0 < phtdoAlt->matHtdo[j][k]) {
            *piInz = MIN(*piInz,
                         (pparms->matNeuTypInzidenzen[i][k]
                         / phtdoAlt->matHtdo[j][k])
                        );
         } /* endif */
      } /* endif */
   } /* endfor */


   /******************************************************************
    * *piInz darf nicht kleiner als iMinimum sein.
    * Falls j der letzte AltTyp von AltTypVorher ist, muss *piInz
    * gleich der Zahl der restlichen NeuTypFahnen sein.
    * Falls i das letzte Exemplar von NeuTypVorher ist, muss *piInz
    * gleich der Zahl der restlichen AltTypFahnen sein.
    ******************************************************************/
   /* fprintf(OUTDAT,"i=%d,j=%d,*piInz=%d,NTFV[ntv]=%d\n",i,j,*piInz,pparms->matNeuTypFahnenVorher[i][pparms->iAltTypVorher]);*/
   if (   *piInz < pparms->iMinimum
      ||  ( j == phtdo->vecAltTypVorherEnde[pparms->iAltTypVorher]
           && *piInz < pparms->matNeuTypFahnenVorher[i][pparms->iAltTypVorher])
      ||  ( i == phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher]
           && *piInz < pparms->matAltTypFahnen[pparms->iNeuTypVorher][j])
      ) {
      bAufgabeGeloest = FALSE;
      *piInz = 0;
   } /* endif */


   /******************************************************************
    * Falls die Aufgabe loesbar war, werden nun die Parms aktualisiert.
    ******************************************************************/
   if ( (bAufgabeGeloest) && (*piInz > 0)) {
      pparms->matAltTypFahnen[pparms->iNeuTypVorher][j] -= *piInz;
      pparms->matNeuTypFahnenVorher[i][pparms->iAltTypVorher] -= *piInz;

      for ( k=0; k<j ; k++) {
         pparms->matAltTypInzidenzen[k][j] -= *piInz * phtdo->matInz[i][k];
      } /* endfor */
      pparms->matAltTypInzidenzen[j][j] -= *piInz * (*piInz -1);

      for ( k=0; k < phtdo->iNeuTypAnzahlVorher ; k++) {
         if (k == pparms->iNeuTypVorher) {
            pparms->matNeuTypInzidenzen[i][k] -= *piInz * (phtdoAlt->matHtdo[j][k] -1);
         } else {
            pparms->matNeuTypInzidenzen[i][k] -= *piInz * phtdoAlt->matHtdo[j][k];
         } /* endif */
      } /* endfor */
   } /* endif */

   if (pparms->bOrdnen) {
      pparms->bOrdnen = ( *piInz == phtdo->matInz[i-1][j] );
   } /* endif */

   /******************************************************************
    * return
    ******************************************************************/
   DBG2("*piInz =%d\n",*piInz);
   TM(lMaxTime += clock() - cl;)
   ++lMaxAnz;
   return bAufgabeGeloest;

} /* bMaximiereInzidenzen */





/******************************************************************
 ******************************************************************
 ** bSucheNiedrigereIndices
 ** Aufgabe dieser Funktion ist es, die Indices der
 ** Inzidenzmatrix solange zurueckzulaufen, bis eine Stelle
 ** gefunden ist, an der eine kleinere Inzidenzanzahl geometrisch
 ** mglich ist. Ab hier kann dann durch Wiederauffuellen der
 ** Matrix die naechste lexikografisch kleinere Loesung gesucht werden.
 ** TRUE  = ein Index (i,j), fr den matInz(i,j) erniedrigt werden konnte,
 **         wurde ermittelt
 ** FALSE = lexikografisch kleinere HTDOs sind nicht mehr moeglich
 ******************************************************************
 ******************************************************************/
static BOOL bSucheNiedrigereIndices(HTDO       * phtdo,
                                    HTDO       * phtdoAlt,
                                    HTDO_PARMS * pparms )

{
   BOOL bLoesbar  = TRUE,
        bGefunden = FALSE;
   INT  *pi       = &pparms->iNeuExemplar,
        *pj       = &pparms->iAltTyp;
   TM(clock_t cl     = clock();)

   DBG2("SucheNiedrigereIndices %ld\n", clock()-cl );
   do {
      /******************************************************************
       * Solange die Inzidenzen == 0 sind, braucht nur heruntergezhlt
       * zu werden.
       ******************************************************************/
      while (bLoesbar && 0==phtdo->matInz[*pi][*pj]) {
         if (*pj >0) {
            /******************************************************************
             * Wenn *pj>0 ist, braucht nur *pj heruntergezhlt zu werden.
             * Falls dabei *pj dem AltTypVorherEnde von (AltTypVorher-1)
             * wird, wechselt der AltTypVorher.
             ******************************************************************/
            --*pj;
            if (0 < pparms->iAltTypVorher) {
               if ( *pj == phtdo->vecAltTypVorherEnde[pparms->iAltTypVorher -1]) {
                  --pparms->iAltTypVorher;
               } /* endif */
            } /* endif */
         } else {
            /******************************************************************
             * Das NeuExemplar *pi muss gewechselt werden, falls moeglich.
             ******************************************************************/
            if ( (0 < pparms->iNeuTypVorherIndex)
                 ||
                 (*pi > phtdo->vecNeuTypVorherAnf[pparms->iNeuTypVorher])
               ) {
               /******************************************************************
                * Wenn *pj==0 ist und *pi gewechselt werden kann,
                * ist *pj neu zu initialisieren.
                ******************************************************************/
               *pj = phtdo->iAltTypAnzahl -1;
               pparms->iAltTypVorher = phtdo->iAltTypAnzahlVorher -1;
               /******************************************************************
                * Wechseln von *pi
                * Falls dabei *pi gleich dem NeuTypVorherAnfang von
                * NeuTypVorher ist, wird NeuTypVorherIndex heruntergezhlt und
                * und NeuTypVorher gewechselt.
                ******************************************************************/
               if ( *pi > phtdo->vecNeuTypVorherAnf[pparms->iNeuTypVorher] ) {
                  --*pi;
               } else {
                  --pparms->iNeuTypVorherIndex;
                  pparms->iNeuTypVorher =
                      pparms->vecNeuTypVorherIndex[pparms->iNeuTypVorherIndex];
                  *pi = phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher];
               } /* endif */
            } else {
               /*****************************************************
                * *pj ==0, wechseln von *pi nicht moeglich;
                * also sind keine niedrigeren Indices mehr da.
                *****************************************************/
               bLoesbar = FALSE;
            } /* endif */
         } /* endif */
      } /* endwhile */

      if (bLoesbar) {
         /******************************************************************
          * Ab hier ist 0 < phtdo->matInz[*pi][*pj]
          ******************************************************************
          * Zunchst wird der Index des ersten AltTyps auf NeuExemplar i ermittelt
          ******************************************************************/
         INT j2 = 0;
         INT k;
         while (phtdo->matInz[*pi][j2] ==0) {
            j2++;
         } /* endwhile */
         /******************************************************************
          * Wenn *pj nicht der erste AltTyp auf NeuExemplar *pi ist
          *   oder er ist es, aber die Zahl der verbliebenen AltTypFahnen
          *      ist so klein im Verhltnis zur Zahl der verbliebenen
          *      NeuTypenVorher, dass sich spter kein NeuExemplar des
          *      NeuTypenVorher mit mehr Inzidenzen ergeben muesste
          *      (d.h. diese Inzidenz ist erstes Sortierfeld aber
          *      die Anzahl der Inzidenzen uebersteigt noch die
          *      Mindestanzahl fuer lexikographisch absteigende
          *      Sortierung)
          * und *pj ist nicht der letzte AltTyp des AltTypVorher
          * und *pi ist nicht das letzte NeuExemplar des NeuTypVorher
          * dann kann die Anzahl der Inzidenzen um 1 erniedrigt werden
          *      und die Aufgabe ist geloest.
          * Sonst muessen alle Inzidenzen geloescht werden und weiter nach
          *      einem anderen Index gesucht werden
          ******************************************************************/
         if (  (j2 < *pj
               || phtdo->matInz[*pi][*pj] > (pparms->matAltTypFahnen[pparms->iNeuTypVorher][*pj]
                                             + phtdo->matInz[*pi][*pj]
                                             + phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher]
                                             - *pi
                                            )
                                            / (phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher]
                                               - *pi +1)
               )
            && ( *pj < phtdo->vecAltTypVorherEnde[pparms->iAltTypVorher])
            && ( *pi < phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher])
            ) {

            /******************************************************************
             * phtdo->matInz[*pi][*pj] um 1 erniedrigen und pparms anpassen
             ******************************************************************/
            phtdo->matInz[*pi][*pj]--;
            pparms->matAltTypFahnen[pparms->iNeuTypVorher][*pj]++;
            pparms->matNeuTypFahnenVorher[*pi][pparms->iAltTypVorher]++;

            for ( k=0; k<*pj ; k++) {
               pparms->matAltTypInzidenzen[k][*pj] += phtdo->matInz[*pi][k];
            } /* endfor */
            pparms->matAltTypInzidenzen[*pj][*pj] += 2 * phtdo->matInz[*pi][*pj];

            for ( k=0; k < phtdo->iNeuTypAnzahlVorher ; k++) {
               if (k == pparms->iNeuTypVorher) {
                  pparms->matNeuTypInzidenzen[*pi][k] += (phtdoAlt->matHtdo[*pj][k] -1);
               } else {
                  pparms->matNeuTypInzidenzen[*pi][k] += phtdoAlt->matHtdo[*pj][k];
               } /* endif */
            } /* endfor */

            bGefunden = TRUE;
            pparms->bOrdnen = FALSE;
         } else {

            /******************************************************************
             * phtdo->matInz[*pi][*pj] Null setzen und pparms anpassen
             ******************************************************************/
            INT iInz = phtdo->matInz[*pi][*pj];
            pparms->matAltTypFahnen[pparms->iNeuTypVorher][*pj] += iInz;
            pparms->matNeuTypFahnenVorher[*pi][pparms->iAltTypVorher] += iInz;

            for ( k=0; k<*pj ; k++) {
               pparms->matAltTypInzidenzen[k][*pj] += phtdo->matInz[*pi][k] * iInz;
            } /* endfor */
            pparms->matAltTypInzidenzen[*pj][*pj] += iInz * (iInz -1);

            for ( k=0; k < phtdo->iNeuTypAnzahlVorher ; k++) {
               if (k == pparms->iNeuTypVorher) {
                  pparms->matNeuTypInzidenzen[*pi][k] += (phtdoAlt->matHtdo[*pj][k] -1) * iInz;
               } else {
                  pparms->matNeuTypInzidenzen[*pi][k] += phtdoAlt->matHtdo[*pj][k] * iInz;
               } /* endif */
            } /* endfor */

            phtdo->matInz[*pi][*pj] =0;
         } /* endif */
      } /* endif */
   } while (bLoesbar && !bGefunden); /* enddo */

   TM(lNiedrigTime += clock() - cl;)
   ++lNiedrigAnz;
   return bLoesbar; /* oder return bGefunden */
} /** end bSucheNiedrigereIndices **/






/******************************************************************
 ******************************************************************
 ** erhoeheIndices
 ******************************************************************
 ******************************************************************/
static void erhoeheIndices(HTDO        * phtdo,
                           HTDO_PARMS  * pparms )

{
   INT  *pi       = &pparms->iNeuExemplar,
        *pj       = &pparms->iAltTyp;
   BOOL bNeuExemplarWechsel = FALSE,
        bWeiter             = FALSE;
   TM(clock_t cl     = clock();)


   DBG2("erhoeheIndices %ld\n",clock()-cl);
   /**********************************************************************
    *  Falls *pj erhoeht werden kann, ist alles einfach.
    **********************************************************************/
   if (*pj < phtdo->iAltTypAnzahl -1) {
      if (*pj == phtdo->vecAltTypVorherEnde[pparms->iAltTypVorher]) {
         ++pparms->iAltTypVorher;
      } /* endif */
      ++*pj;
   } else {
      /**********************************************************************
       *  *pi muss erhoeht werden
       **********************************************************************/
      *pj = 0;
      pparms->iAltTypVorher = 0;

      bNeuExemplarWechsel = TRUE;
      if (*pi == phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher]) {
         ++pparms->iNeuTypVorherIndex;
         pparms->iNeuTypVorher
              = pparms->vecNeuTypVorherIndex[pparms->iNeuTypVorherIndex];
         *pi =phtdo->vecNeuTypVorherAnf[pparms->iNeuTypVorher];
         /* fprintf(OUTDAT,"ntv:%d\n",pparms->iNeuTypVorher);*/
         /**********************************************************************
          *  Nach Wechsel des NeuTypVorher muss nicht lexikografisch
          *  geordnet werden, also Ordnen=FALSE.
          **********************************************************************/
         pparms->bOrdnen = FALSE;
      } else {
         ++*pi;
         /**********************************************************************
          *  Der NeuTypVorher wurde nicht gewechselt, also darf
          *  das NeuExemplar lexikografisch nicht groesser als
          *  das letzte werden.
          **********************************************************************/
         pparms->bOrdnen = TRUE;
      } /* endif */

   } /* endif */

   /**********************************************************************
    *  Solange die Fahnen = 0 sind, kann matInz sofort = 0 gesetzt werden.
    *  Ausnahme: Der AltTypVorher aendert sich und die zugehrigen
    *  NeuTypFahnenVorher sind noch > 0. Dann wird MaximiereInzidenzen
    *  auf HtdoNichtMoeglich erkennen.
    **********************************************************************/
   bWeiter = (*pj < phtdo->iAltTypAnzahl -1
             && 0 == pparms->matAltTypFahnen[pparms->iNeuTypVorher][*pj]);
   while (bWeiter) {
      if (*pj == phtdo->vecAltTypVorherEnde[pparms->iAltTypVorher]) {
         if (0 < pparms->matNeuTypFahnenVorher[*pi][pparms->iAltTypVorher]) {
            bWeiter = FALSE;
         } else {
            ++pparms->iAltTypVorher;
         } /* endif */
      } /* endif */
      if (bWeiter) {
         phtdo->matInz[*pi][*pj] = 0;
         if (pparms->bOrdnen) {
            pparms->bOrdnen = (phtdo->matInz[*pi][*pj] == phtdo->matInz[*pi -1][*pj]);
         } /* endif */
         ++*pj;
         bWeiter = (*pj < phtdo->iAltTypAnzahl -1
                   && 0 == pparms->matAltTypFahnen[pparms->iNeuTypVorher][*pj]);
      } /* endif */
   } /* endwhile */

   /**********************************************************************
    *  Beim Wechsel des NeuTypExemplars ist *pj nun der erste AltTyp, zu
    *  dem es Inzidenzen gibt. Die Anzahl der AltTypFahnen darf mit
    *  steigendem *pi nicht zunehmen, da die Geometrien lexikografisch
    *  geordnet werden muessen. Daraus ergibt sich ein Mindestwert fuer
    *  matInz[*pi][*pj].
    *  Dieses Minimum ist
    *  (Anzahl der vebliebenen Fahnen) div (Anzahl der verbl. NeuTypenVorher)
    *  + (1 falls Fahnen mod NeuTypenVorher >= 1).
    *  Dies entspricht der folgenden Formel.
    **********************************************************************/
   if (bNeuExemplarWechsel) {
      pparms->iMinimum = (pparms->matAltTypFahnen[pparms->iNeuTypVorher][*pj]
                          + phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher]
                          - *pi)
                         / (phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher]
                            - *pi +1);
   } else {
      pparms->iMinimum = 0;
   } /* endif */
   TM(lErhoeheTime += clock() - cl;)
   ++lErhoeheAnz;

} /* erhoeheIndices */





/******************************************************************
 ******************************************************************
 ** berechneNaechsteHtdo
 ******************************************************************
 ******************************************************************/
static void berechneNaechsteHtdo(HTDO* phtdoAlt)
{
   INT         i, iAltTypVorher, iSumme;
   LONG        lDurchlauf = 0;
   LONG        lSuchnext  = 0;
   HTDO*       phtdo     = (HTDO*) calloc (sizeof(HTDO), 1);
   HTDO_PARMS* pparms;
   BOOL        bHtdoMoeglich;
   BOOL        bHtdoGefunden = FALSE;


   DBG2_("berechneNaechsteHtdo\n");

   /******************************************************************
    ** Zunaechst wird die HTDO initialisiert.
    ******************************************************************/
   phtdo->iNeuAnzahl = phtdoAlt->iAltAnzahl;
   phtdo->iAltAnzahl = phtdoAlt->iNeuAnzahl;

   phtdo->iAltTypAnzahlVorher = phtdoAlt->iNeuTypAnzahlVorher;
   phtdo->iNeuTypAnzahlVorher = phtdoAlt->iAltTypAnzahl;
   phtdo->iAltTypAnzahl       = phtdoAlt->iNeuTypAnzahl;

   phtdo->iStufe              = phtdoAlt->iStufe +1;

   memcpy(phtdo->vecAltVorher, phtdoAlt->vecNeuVorher, sizeof(VECTOR));
   memcpy(phtdo->vecNeuVorher, phtdoAlt->vecAlt, sizeof(VECTOR));
   memcpy(phtdo->vecAlt      , phtdoAlt->vecNeu, sizeof(VECTOR));

   phtdo->phtdoAlt            = phtdoAlt;


   /******************************************************************
    *   Initialisieren des Vectors der AltTypVorherEnden
    ******************************************************************/
   iAltTypVorher = 0;
   iSumme = 0;
   for ( i = 0; i < phtdo->iAltTypAnzahl; ++i) {
     iSumme += phtdo->vecAlt[i];
     if (iSumme == phtdo->vecAltVorher[iAltTypVorher]) {
        phtdo->vecAltTypVorherEnde[iAltTypVorher] = i;
        iSumme = 0;
        ++iAltTypVorher;
     } /* endif */
   } /* endfor */

   /******************************************************************
    *   Initialisieren des Vectors der NeuTypVorherEnden
    ******************************************************************/
   phtdo->vecNeuTypVorherEnde[0] = phtdo->vecNeuVorher[0] -1;
   for ( i = 1; i < phtdo->iNeuTypAnzahlVorher; ++i) {
      phtdo->vecNeuTypVorherEnde[i] = phtdo->vecNeuTypVorherEnde[i-1]
                                         + phtdo->vecNeuVorher[i];
   } /* endfor */

   /******************************************************************
    *   Initialisieren des Vectors der NeuTypVorherAnfaenge
    ******************************************************************/
   phtdo->vecNeuTypVorherAnf[0] = 0;
   for ( i = 1; i < phtdo->iNeuTypAnzahlVorher; ++i) {
      phtdo->vecNeuTypVorherAnf[i] = phtdo->vecNeuTypVorherEnde[i-1] +1;
   } /* endfor */




   DBG1("berechneNaechsteHtdo2\n");
   /******************************************************************
    ** Initialisierung der PARMS
    ******************************************************************/
   pparms = (HTDO_PARMS*) calloc (sizeof(HTDO_PARMS), 1);
   InitialisiereParms(phtdo, phtdoAlt, pparms);
   DBG_F(DebugHtdo(phtdo));

   /******************************************************************
    *   Hier folgt der eigentliche Algorithmus zur Errechnung der
    *   HTDOs.
    *   Die HTDOs werden in lexikografischer Reihenfolge erzeugt.
    *   Daher muss zu jedem Index (i,j) zunaechst der denkbar groesste
    *   Wert gefunden werden, bis der letzte Index erreicht und damit die
    *   HTDO erzeugt ist. In diesem Fall werden die HTDOs der nchsten
    *   Stufe - und alle darauf aufbauenden (H)TDOs tieferer Stufen -
    *   ermittelt (es sei denn, die HTDO ist bereits eine TDO).
    *   Konnte zu einem Index kein plausibler InzidenzWert gefunden werden
    *   oder zu einer HTDO wurden alle TDOs ermittelt, so wird der
    *   letzte vorhergehende Index gesucht, dessen Wert erniedrigt werden
    *   kann. Darauf aufbauend werden dann wieder die Indices erhoeht
    *   und deren maximale Inzidenzwerte ermittelt.
    *   Dies endet, wenn kein Index mehr gefunden wurde, dessen Wert
    *   erniedrigt werden konnte, denn damit sind keine lexikografisch
    *   kleineren HTDOs mehr mglich.
    ******************************************************************/
   do {
      bHtdoMoeglich = bMaximiereInzidenzen(phtdo, phtdoAlt, pparms );
      ++lDurchlauf;
      if (lDurchlauf == 1) {
         lDbg = 1;
      } /* endif */
      if (0 == lDurchlauf % 100000) {
         fprintf(OUTDAT,".");
         DBG2_2("lDurchlauf=%ld\n",lDurchlauf);
         DBG2_F(DebugHtdo(phtdo));
      } /* endif */
      if ( /* HTDO moeglich und fertig */ bHtdoMoeglich
           && pparms->iNeuTypVorherIndex == phtdo->iNeuTypAnzahlVorher -1
           && pparms->iNeuExemplar == phtdo->vecNeuTypVorherEnde[pparms->iNeuTypVorher]
           && pparms->iAltTyp      == phtdo->iAltTypAnzahl -1) {
         if ( bNormalisierteHtdoIstTdo(phtdo) ) {
            ++vecTestfall[phtdo->iStufe];
            Ausgabe(phtdo,TRUE);
         } else {
            if (bPruefeMaxfit(phtdo) && bPruefeMaxfit2(phtdo) && bPruefeInzidenzen(phtdo)) {
               DBG3("\n MAxStufe=%d,htdo-Stufe=%d\n",iMaxStufe,phtdo->iStufe);
               ++vecTestfall[phtdo->iStufe];
               Ausgabe(phtdo,FALSE);
               if (iMaxStufe > phtdo->iStufe) {
                  berechneNaechsteHtdo(phtdo);
               } /* endif */
            } /* endif */
         } /* endif */
         bHtdoMoeglich = FALSE;
         bHtdoGefunden = TRUE;
      } /* endif */
      if ( !bHtdoMoeglich ) {
         DBG2( "%ld.",lDurchlauf);
         ++lSuchnext;
         bHtdoMoeglich = bSucheNiedrigereIndices( phtdo, phtdoAlt, pparms );
         /* if (phtdo->iStufe == 4 && lSuchnext % 1 == 0 ) {
            fprintf(OUTDAT, "such %d. %d ",lSuchnext,clock()-cl);
            DebugHtdo(phtdo);
            fflush(OUTDAT);*/
         /*}  endif */
      }
      if ( bHtdoMoeglich ) {
         erhoeheIndices(phtdo, pparms );
         DBG_F(AusgabeParms(pparms,phtdo));
      } /* endif */
   } while ( bHtdoMoeglich ); /* enddo */

   /******************************************************************
    *   Abschlussarbeiten
    ******************************************************************/
   fprintf(OUTDAT, "\nEnde  ");   /*DB  Wort Testfall weggenommen*/
   for (i=1; i<phtdo->iStufe ; ++i) {
      fprintf(OUTDAT, "%d.", vecTestfall[i]);
   } /* endfor */
   fprintf(OUTDAT, ": %ld clocks",clock()-cl);
   vecTestfall[phtdo->iStufe] =0;
   free(pparms);
   free(phtdo);

} /* berechneNaechsteHtdo */






/******************************************************************
 ******************************************************************
 ** Initialisiere_Parms
 ******************************************************************
 ******************************************************************/
static void Initialisiere_Parms(GERADENFALL gf,HTDO* phtdo, HTDO1_PARMS* pparms)
{
   INT i, i2, j;

   DBG1("Initialisiere_Parms\n");

   /******************************************************************
    * evtl. 2-Geraden streichen
    * Die Zahl der Verbindungen durch 2-Geraden wird 'Zweier' genannt.
    ******************************************************************/
   if ( 2 == gf.agt[gf.iGeradenTypAnzahl-1].iLaenge ) {
      pparms->iZweier = gf.agt[gf.iGeradenTypAnzahl-1].iAnzahl * 2;
      phtdo->iAltTypAnzahl--;
      phtdo->iAltAnzahl -= gf.agt[gf.iGeradenTypAnzahl -1].iAnzahl;
   } else {
      pparms->iZweier = 0;
   } /* endif */

   /******************************************************************
    *   Initialisieren der Matrix der Geraden-Schneidungen
    *   Sie enthlt die Anzahl der Inzidenzen, die zwischen den
    *   Geradentypen i und i2 bestehen drfen (bzw. das doppelte dieser
    *   Anzahl bei i = i2 zur Vereinfachung der Rechnung)
    ******************************************************************/
   for ( i = 0; i < phtdo->iAltTypAnzahl; ++i) {
      pparms->matGerSchneidungen[i][i] = gf.agt[i].iAnzahl * (gf.agt[i].iAnzahl - 1);
      for ( i2 = i+1; i2 < phtdo->iAltTypAnzahl; ++i2)
         pparms->matGerSchneidungen[i][i2] = gf.agt[i].iAnzahl * gf.agt[i2].iAnzahl;
   }


   /******************************************************************
    *   Initialisieren des Fahnen-Vectors
    *   Er enthlt die Anzahl der Inzidenzen der einzelnen
    *   Geradentypen
    ******************************************************************/
   for ( j = 0; j < phtdo->iAltTypAnzahl; ++j)
      pparms->vecFahnen[j] = gf.agt[j].iAnzahl * gf.agt[j].iLaenge;


   /******************************************************************
    *   Initialisieren des Vectors der Punkte-Verbindungen
    *   Er enthlt zu jedem Punkt die Anzahl der Punkte, mit denen
    *   er zu verbinden ist
    ******************************************************************/
   for ( i = 0; i < gf.iPunkteAnzahl; ++i)
      pparms->vecPktVerbindungen[i] = gf.iPunkteAnzahl - 1;

   /******************************************************************
    *   Initialisieren von iMinimum
    ******************************************************************/
   pparms->iMinimum = (pparms->vecFahnen[0] + gf.iPunkteAnzahl -1)
                      / (gf.iPunkteAnzahl);

   /******************************************************************
    *   Initialisieren von bOrdnen
    ******************************************************************/
   pparms->bOrdnen  = FALSE;


   return;
} /* Initialisiere_Parms */





/******************************************************************
 ******************************************************************
 ** bMaximiereGeradeAufPunkt
 ** TRUE  = der maximale Wert von matInz(i,j) konnte ermittelt werden
 ** FALSE = jeder denkbare Wert fhrt zu Widersprchen
 ******************************************************************
 ******************************************************************/
static BOOL bMaximiereGeradeAufPunkt(INT           i, INT j,
                                     HTDO        * phtdo,
                                     HTDO1_PARMS * pparms,
                                     GERADENTYP    agt[] )

{
   BOOL  bAufgabeGeloest = TRUE;
   INT * piInz = &phtdo->matInz[i][j];
   INT   k;

   DBG1("bMaximiereGeradeAufPunkt\n");

   /******************************************************************
    * Die Anzahl der Inzidenzen zwischen dem i-ten Punkt und dem
    * dem j-ten Geradentyp wird initialisiert mit dem Maximum aus
    * der Anzahl der Fahnen und der Anzahl der Geraden des Geradentyps
    ******************************************************************/
   *piInz = MIN(pparms->vecFahnen[j], phtdo->vecAlt[j]);

   DBG2("*piInz1 = %d\n",*piInz);
   /******************************************************************
    * Falls geordnet werden muss, begrenzt der (i-1)-Punkt die Inzidenzen
    ******************************************************************/
   if (pparms->bOrdnen) {

      *piInz = MIN(*piInz, phtdo->matInz[i-1][j]);
   } /* endif */

   /******************************************************************
    * Die Schneidungen der Geradentypen sind durch matGerSchneidungen begrenzt *
    ******************************************************************/
   for ( k=0; k<j ; k++) {
      if (0 < phtdo->matInz[i][k]) {
         *piInz = MIN(*piInz, (pparms->matGerSchneidungen[k][j] / phtdo->matInz[i][k]) );
      } /* endif */
   } /* endfor */
   while (*piInz * (*piInz -1) > pparms->matGerSchneidungen[j][j]) {
      --*piInz;
   } /* endwhile */

   DBG2("*piInz2 = %d\n",*piInz);

   /******************************************************************
    * Der Punkt i wird (*piInz)-mal mit weiteren (Geradenlaenge-1)
    * Punkten verbunden. Die Verbindungen zwischen den Punkten sind
    * durch vecPktVerbindungen begrenzt.
    ******************************************************************/
   *piInz = MIN(*piInz, (pparms->vecPktVerbindungen[i] / (agt[j].iLaenge -1)) );

   DBG2("*piInz3 = %d\n",*piInz);
   /******************************************************************
    * *piInz darf nicht kleiner als iMinimum sein.
    * Falls j der letzte Geradentyp ist, darf die Zahl der Zweier
    * nicht kleiner sein als die Zahl der restlichen PktVerbindungen.
    * Falls i der letzte Punkt ist, duerfen keine Fahnen uebrig sein.
    ******************************************************************/
   if (   *piInz < pparms->iMinimum
      ||  (  j == phtdo->iAltTypAnzahl -1
          && pparms->iZweier < pparms->vecPktVerbindungen[i]
                               - *piInz * (agt[j].iLaenge -1)
          )
      ||  (  i == (phtdo->vecNeuVorher[0] -1) && *piInz < pparms->vecFahnen[j] )
      ) {
      bAufgabeGeloest = FALSE;
      *piInz = 0;
   }

   /******************************************************************
    * Falls die Aufgabe loesbar war, werden nun die Parms aktualisiert.
    ******************************************************************/
   if ( (bAufgabeGeloest) && (*piInz > 0)) {
      pparms->vecPktVerbindungen[i] -= *piInz * (agt[j].iLaenge -1);
      pparms->vecFahnen[j]          -= *piInz;
      for ( k = 0; k<j ; k++) {
         pparms->matGerSchneidungen[k][j] -= *piInz * phtdo->matInz[i][k];
      } /* endfor */
      pparms->matGerSchneidungen[j][j] -= *piInz * (*piInz -1);
   } /* endif */
   if (pparms->bOrdnen) {
      pparms->bOrdnen = ( *piInz == phtdo->matInz[i-1][j] );
   } /* endif */

   DBG2("*piInz4 = %d\n",*piInz);

   /******************************************************************
    * return
    ******************************************************************/
   return bAufgabeGeloest;

} /* bMaximiereGeradeAufPunkt */






/******************************************************************
 ******************************************************************
 ** bSuche_NiedrigereIndices
 ** Aufgabe dieser Funktion ist es, die Indices *pi und *pj der
 ** Inzidenzmatrix solange zurueckzulaufen, bis eine Stelle
 ** gefunden ist, an der eine kleinere Inzidenzanzahl geometrisch
 ** mglich ist. Ab hier kann dann durch Wiederauffuellen der
 ** Matrix die naechste lexikografisch kleinere Loesung gesucht werden.
 ** TRUE  = ein Index (i,j), fr den matInz(i,j) erniedrigt werden konnte,
 **         wurde ermittelt
 ** FALSE = lexikografisch kleinere HTDOs sind nicht mehr moeglich
 ******************************************************************
 ******************************************************************/
static BOOL bSuche_NiedrigereIndices(INT         * pi,
                                     INT         * pj,
                                     HTDO        * phtdo,
                                     HTDO1_PARMS * pparms,
                                     GERADENTYP    agt[]  )

{
   BOOL bLoesbar  = TRUE,
        bGefunden = FALSE;
   INT  j2, k2;


   DBG1("bSuche_NiedrigereIndices\n");

   do {
      /******************************************************************
       * Solange die Inzidenzen == 0 sind, braucht nur heruntergezhlt
       * zu werden.
       ******************************************************************/
      while (bLoesbar && 0==phtdo->matInz[*pi][*pj]) {
         if (*pj >0) {
            (*pj)--;
         } else {
            if (*pi >0){
               *pj = phtdo->iAltTypAnzahl -1;
               (*pi)--;
               pparms->iZweier += pparms->vecPktVerbindungen[*pi];
            } else {
               /*****************************************************
                * *pi == *pj ==0,
                * also sind keine niedrigeren Indices mehr da.
                *****************************************************/
               bLoesbar = FALSE;
            } /* endif */
         } /* endif */
      }; /* endwhile */

      if (bLoesbar) {
         /******************************************************************
          * Ab hier ist 0 < phtdo->matInz[*pi][*pj]
          ******************************************************************
          * Zunchst wird der Index des ersten Geradentyps auf Punkt i ermittelt
          ******************************************************************/
         for ( j2 =0; phtdo->matInz[*pi][j2] ==0; ++j2) {
            ;
         } /* endfor */

         /******************************************************************
          * Wenn *pj nicht der erste Geradentyp auf Punkt i ist
          *   oder er ist es, aber die Zahl der Fahnen ist so klein im
          *      Verhltnis zur Zahl der verbliebenen Punkte, dass sich
          *      spter kein Punkt mit mehr Inzidenzen ergeben muesste (d.h.,
          *      die Punkte koennen noch lexikographisch absteigend
          *      sortiert werden)
          * und *pj ist nicht der letzte Geradentyp
          *   oder er ist es, aber die Anzahl der Verbindungen, die durch
          *      die Zweiergeraden hergestellt werden kann, ist groesser
          *      als die Anzahl der noch herzustellende Punktverbindungen
          * dann kann die Anzahl der Inzidenzen um 1 erniedrigt werden
          *      und die Aufgabe ist geloest.
          * Sonst muessen alle Inzidenzen geloescht werden und weiter nach
          *      einem anderen Index gesucht werden
          ******************************************************************/
         if (  (j2 < *pj
               || phtdo->matInz[*pi][*pj] > (pparms->vecFahnen[*pj]
                                             + phtdo->matInz[*pi][*pj]
                                             + phtdo->vecNeuVorher[0]
                                             - *pi
                                            )
                                            / (phtdo->vecNeuVorher[0] - *pi +1)
               )
            && ( *pj < phtdo->iAltTypAnzahl -1
               || pparms->iZweier >= pparms->vecPktVerbindungen[*pi]
                                     + agt[*pj].iLaenge -1
               )
            ) {
            /******************************************************************
             * phtdo->matInz[*pi][*pj] um 1 erniedrigen und pparms anpassen
             ******************************************************************/
            phtdo->matInz[*pi][*pj]--;
            pparms->vecPktVerbindungen[*pi] += agt[*pj].iLaenge -1;
            pparms->vecFahnen[*pj]++;
            for ( k2 =0; k2 < *pj; k2++ ) {
               pparms->matGerSchneidungen[k2][*pj] += phtdo->matInz[*pi][k2];
            }; /* endfor */
            pparms->matGerSchneidungen[*pj][*pj] += 2 * phtdo->matInz[*pi][*pj];
            bGefunden = TRUE;
            pparms->bOrdnen = FALSE;
         } else {
            /******************************************************************
             * phtdo->matInz[*pi][*pj] Null setzen und pparms anpassen
             ******************************************************************/
            pparms->vecPktVerbindungen[*pi] += (agt[*pj].iLaenge -1)
                                                  * phtdo->matInz[*pi][*pj] ;
            pparms->vecFahnen[*pj] += phtdo->matInz[*pi][*pj];
            for ( k2 =0; k2 < *pj; k2++ ) {
               pparms->matGerSchneidungen[k2][*pj] += (phtdo->matInz[*pi][k2]) *
                                                      (phtdo->matInz[*pi][*pj]);
            } /* endfor */
            pparms->matGerSchneidungen[*pj][*pj] += phtdo->matInz[*pi][*pj] *
                                                   (phtdo->matInz[*pi][*pj] -1);
            phtdo->matInz[*pi][*pj] =0;
         }; /* endif */
      }; /* endif */
   } while (bLoesbar && !bGefunden); /* enddo */

   /******************************************************************
    * return
    ******************************************************************/
   return bLoesbar; /* oder return bGefunden */

} /** end bSuche_NiedrigereIndices **/







/******************************************************************
 ******************************************************************
 ** erhoehe_Indices
 ******************************************************************
 ******************************************************************/
static void erhoehe_Indices(INT         * pi, INT * pj,
                           HTDO        * phtdo,
                           HTDO1_PARMS * pparms )

{
   DBG1("erhoehe_Indices\n");


   /**********************************************************************
    *  Falls *pj erhoeht werden kann, ist alles einfach.
    **********************************************************************/
   if (*pj < phtdo->iAltTypAnzahl -1) {
      ++*pj;
      pparms->iMinimum = 0;
      phtdo->matInz[*pi][*pj] = 0;
   } else {
      /**********************************************************************
       *  *pi muss erhoeht werden
       **********************************************************************
       *  Die Zweier sind um die Anzahl der noch nicht gezogenen Verbindungen
       *  zu verringern.
       **********************************************************************/
      pparms->iZweier -= pparms->vecPktVerbindungen[*pi];
      *pj = 0;
      ++*pi;
      phtdo->matInz[*pi][*pj] = 0;

      /**********************************************************************
       *  Der neue Punkttyp darf lexikografisch nicht groesser als der alte
       *  sein, also Ordnen=TRUE.
       **********************************************************************/
      pparms->bOrdnen = TRUE;

      /**********************************************************************
       *  Solange die Fahnen = 0 sind, kann matInz sofort = 0 gesetzt werden.
       **********************************************************************/
      while (*pj < phtdo->iAltTypAnzahl -1 && 0 == pparms->vecFahnen[*pj]) {
         phtdo->matInz[*pi][*pj] = 0;
         if (pparms->bOrdnen) {
            pparms->bOrdnen = (phtdo->matInz[*pi][*pj] == phtdo->matInz[*pi -1][*pj]);
         } /* endif */
         ++*pj;
      } /* endwhile */
      /**********************************************************************
       *  Die Anzahl der Typ-j-Geraden darf mit steigendem i nicht zunehmen,
       *  da die Geometrien lexikografisch geordnet werden muessen.
       *  Die Anzahl der verbliebenen Punkte ist phtdo->iNeuAnzahl - *pi.
       *  Das Minimum fuer matInz[*pi][*pj] ist
       *  (Anzahl der Fahnen) div (Anzahl der verbl. Punkte)
       *  + (1 falls Fahnen mod Punkte >= 1).
       *  Dies entspricht der folgenden Formel.
       **********************************************************************/
      pparms->iMinimum = (pparms->vecFahnen[*pj] + phtdo->iNeuAnzahl - *pi -1)
                         / (phtdo->iNeuAnzahl - *pi);
   } /* endif */

} /* erhoehe_Indices */






/******************************************************************
 ******************************************************************
 ** berechneErsteHtdo
 ******************************************************************
 ******************************************************************/
static void berechneErsteHtdo(GERADENFALL gf)
/* berechnet Punkttypenverteilung */
{
  /* DIE GERADEN MUESSEN SORTIERT SEIN, DIE GROESSTE GERADE ZUERST */
  HTDO        *phtdo;
  HTDO1_PARMS *pparms;

  /* i und j sind die Indices fuer Punkttyp und Geradentyp */
  INT  i = 0;
  INT  j = 0;
  BOOL bHtdoMoeglich = TRUE;
  LONG l1 = 0;
  LONG l2 = 0;


  DBG1("berechneErsteHtdo\n");


  /******************************************************************
   * Testfall setzen
   ******************************************************************/
  ++vecTestfall[1];
  vecTestfall[2] =0;

  /******************************************************************
   * Normalisiere den Geradenfall (nicht vorhandene Geraden streichen)
   ******************************************************************/
  gf.iGeradenAnzahl = 0;
  for ( l2=0; l2<gf.iGeradenTypAnzahl; ++l2) {
     if (0 < gf.agt[l2].iAnzahl) {
        gf.agt[l1] = gf.agt[l2];
        gf.iGeradenAnzahl += gf.agt[l1].iAnzahl;
        ++l1;
     }  /* endif */
  } /* endfor */
  gf.iGeradenTypAnzahl = l1;


  /******************************************************************
   * Sonderfall nur 2-Geraden
   ******************************************************************/
  if ( 1 == gf.iGeradenTypAnzahl && 2 == gf.agt[0].iLaenge) {
     fprintf(OUTDAT, " Testfall 1.");
     fprintf(OUTDAT, "\n TDO auf Stufe 1\n");
     fprintf(OUTDAT, " Es gibt  0 Geradentypen und 1 Punkttyp\n" );
     fprintf(OUTDAT, "      %3d: \n", gf.iPunkteAnzahl);

     lLoesungsAnzahl++;
     return;
  }  /* endif */


  /******************************************************************
   * Hier beginnt die Initialisierung der Variablen.
   ******************************************************************/
  phtdo = (HTDO*) calloc(sizeof(HTDO),1);
  phtdo->iNeuAnzahl          = gf.iPunkteAnzahl;
  phtdo->iAltAnzahl          = gf.iGeradenAnzahl;
  for (l1 = 0; l1 < gf.iGeradenTypAnzahl ; ++l1) {
     phtdo->vecAlt[l1] = gf.agt[l1].iAnzahl;
  } /* endfor */
  phtdo->iAltTypAnzahl       = gf.iGeradenTypAnzahl;
  phtdo->vecNeuVorher[0]     = gf.iPunkteAnzahl;
  phtdo->iNeuTypAnzahlVorher = 1;
  phtdo->iStufe              = 2;

  pparms = (HTDO1_PARMS*) calloc(sizeof(HTDO1_PARMS),1);

  Initialisiere_Parms(gf, phtdo, pparms);


  /******************************************************************
   *   Hier folgt der eigentliche Algorithmus zur Errechnung der
   *   HTDO_1.
   *   Die HTDOs werden in lexikografischer Reihenfolge erzeugt.
   *   Daher muss zu jedem Index (i,j) zunaechst der denkbar groesste
   *   Wert gefunden werden, bis der letzte Index erreicht und damit die
   *   HTDO erzeugt ist. In diesem Fall werden die HTDOs der nchsten
   *   Stufe (HTDO_2) - und alle darauf aufbauenden (H)TDOs tieferer Stufen -
   *   ermittelt (es sei denn, die HTDO ist bereits eine TDO).
   *   Konnte zu einem Index kein plausibler InzidenzWert gefunden werden
   *   oder zu einer HTDO_1 wurden alle TDOs ermittelt, so wird der
   *   letzte vorhergehende Index gesucht, dessen Wert erniedrigt werden
   *   kann. Darauf aufbauend werden dann wieder die Indices erhoeht
   *   und deren maximale Inzidenzwerte ermittelt.
   *   Dies endet, wenn kein Index mehr gefunden wurde, der erniedrigt
   *   werden konnte, denn damit sind keine lexikografisch kleineren
   *   HTDOs mehr mglich.
   ******************************************************************/
  do {
     DBG_F(Debug_(i,j,phtdo,pparms));
     bHtdoMoeglich = bMaximiereGeradeAufPunkt( i, j, phtdo, pparms, gf.agt );
     if ( /* HTDO moeglich und fertig */ bHtdoMoeglich && i == phtdo->vecNeuVorher[0]-1 && j == phtdo->iAltTypAnzahl-1) {
         if ( bNormalisierteHtdoIstTdo(phtdo) ) {
            ++vecTestfall[phtdo->iStufe];
            Ausgabe(phtdo,TRUE);
         } else {
            if (bTdoPlausi(phtdo,&gf) && bPruefeMaxfit(phtdo)&& bPruefeMaxfit2(phtdo)) {
               ++vecTestfall[phtdo->iStufe];
               Ausgabe(phtdo,FALSE);
               if (iMaxStufe > phtdo->iStufe) {
                  berechneNaechsteHtdo(phtdo);
               } /* endif */
            } /* endif */
         } /* endif */
         bHtdoMoeglich = FALSE;
     }
     if ( !bHtdoMoeglich ) {
         bHtdoMoeglich = bSuche_NiedrigereIndices( &i, &j, phtdo, pparms, gf.agt );
     }
     if ( bHtdoMoeglich ) {
        erhoehe_Indices( &i, &j, phtdo, pparms );
     } /* endif */
  } while ( bHtdoMoeglich ); /* enddo */


  /******************************************************************
   *   Abschlussarbeiten
   ******************************************************************/
  fprintf(OUTDAT, "\nEnde  ");     /* DB Wort Testfall weggenommen*/
  for (i=1; i<phtdo->iStufe ; ++i) {
     fprintf(OUTDAT, "%d.", vecTestfall[i]);
  } /* endfor */
  fprintf(OUTDAT, ": %ld clocks",clock()-cl);
  vecTestfall[phtdo->iStufe] =0;
  free(phtdo);
  free(pparms);

} /* ErsteHtdo */






int Dirk_Kaempfer_main(int argc, char *argv[], char *envp[])
{  /* HAUPTPROGRAMM */
   BOOL        bLoesungVorhanden             = FALSE;
   INT         i;
   GERADENFALL gf;
   time_t      t  = time(NULL);

   cl = clock();



   oeffneDateien();
   InitMaxfit();

   gf = gfLesenEingabe();

   if (bBearbeiteTestfall) {
      HTDO htdo;
      for(i=1; i<=iTestfallAnzahl; ++i) {
         if (bEinlesenTestfall(&htdo,i)) {
            berechneNaechsteHtdo(&htdo);
         } else {
            fprintf(OUTDAT, "\nEinzulesenden Testfall nicht gefunden!");
         }
      }
   } else {
      DBG2("gf.bGeradenAnzahlProTypVorgegeben = %d\n",
             gf.bGeradenAnzahlProTypVorgegeben);
      if ( !gf.bGeradenAnzahlProTypVorgegeben ) {
         /* GeradenTypAnzahlen wurden nicht vorgegeben */
         BildeGleichung(&gf);
         bLoesungVorhanden = bGibNaechsteGleichungsloesung(&gf);
      } else {
         bLoesungVorhanden = TRUE;  /* = bGeradenAnzahlProTypVorgegeben */
      }; /* endif */

      for (;bLoesungVorhanden ;
            bLoesungVorhanden = (!gf.bGeradenAnzahlProTypVorgegeben)
                                && bGibNaechsteGleichungsloesung(&gf)) {
         if ( bBraunbedingungErfuellt(&gf)) {
            AusgabeGeradenwechsel(&gf);
            berechneErsteHtdo(gf);
         } else {
            DBG_F(AusgabeGeradenwechsel(&gf));
         } /* endif */
      } /* endfor */

      fprintf(OUTDAT, "\ninsgesamt %ld Loesungen", lLoesungsAnzahl);
   } /* endif */

   fprintf(OUTDAT, "\nBenoetigte Zeit:  %.0f sec bzw.%ld clocks \n",
                    difftime(time(NULL),t),clock()-cl);
   TM(fprintf(OUTDAT, "Davon innerhalb der Funktionen \n MaximiereInzidenzen %d "
                   "\n ErhoeheIndices %d \n SucheNiedrigereIndices %d \n ",
                    lMaxTime, lErhoeheTime, lNiedrigTime);)
   fprintf(OUTDAT, "Anzahl Aufrufe:\n MaximiereInzidenzen %ld "
                   "\n ErhoeheIndices %ld \n SucheNiedrigereIndices %ld \n "
                   "Initialisieren %ld \n",
                    lMaxAnz, lErhoeheAnz, lNiedrigAnz, lInitAnz);

   schliesseDateien();

   return 0;

} /* end main */

#ifdef COMP_START
#endif /* COMP_START */

/* End. */


}}}


