%option interactive
%option noyywrap 

%{
	#include <stdexcept>
	#include <string>
	#include "parser.tab.hpp"
%}

%%

"+" {
	return PLUS; 
}

"-" {
	return MINUS;
}

"*" {
	return TIMES;
}

"^" {
	return EXPONENT;
}

[a-zA-Z_][a-zA-Z0-9_]* {
	yylval.id = yytext;
	return IDENTIFIER;
}

[0-9]+ {
	yylval.num = atoi(yytext);
	return NUMBER;
}

[ \t\r\f]* { } /* Do nothing for newlines. This is also the EOL token */

[\\][ \t\r]+ { /* backslash followed by 1 or more space, \t, \r */
	throw std::invalid_argument("'\\' cannot be followed by space or carrage return."); 
}

[\\] {
	throw std::invalid_argument("Backslashes '\\' not supported."); 
}

"(" { 
	return LPARENTHESIS;
}

")" { 
	return RPARENTHESIS;
}

. { /* Matches everything else */
	throw std::invalid_argument( "Encountered unidentified token: ['" + std::string(yytext) + "']" ); 
}

%%
