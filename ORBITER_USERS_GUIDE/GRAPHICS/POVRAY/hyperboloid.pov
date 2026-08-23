//Brian Vincent
//Hyperboloid

//Files with predefined colors and textures
#include "colors.inc"
#include "glass.inc"
#include "golds.inc"
#include "metals.inc"
#include "stones.inc"
#include "woods.inc"

//Place the camera
camera {
   sky <0,0,1>          //Don't change this
   direction <-1,0,0>   //Don't change this
   right <-4/3,0,0>     //Don't change this
   location  <15,15,5>  //Camera location
   look_at   <0,0,0>    //Where camera is pointing
   angle 22      //Angle of the view
}

//Ambient light to "brighten up" darker pictures
//global_settings { ambient_light White }
global_settings { max_trace_level 10 }


//Place a light
//light_source { <15,30,1> color White*2 }             
//light_source { <10,10,0> color White*2 }             
light_source { <10,0,0> color White*2 }             
light_source { <0,0,10> color White }
//light_source { <0,10,0> color White*2}


      
plane{<0,0,-1>,7 texture {T_Silver_3A}}

//Set a background color
//background { color White }



//Create the array to collect the projected points
#declare b = 16;
#declare q = array[b];
#declare p = array[b];   
#declare s = array[b];
#declare o = array[b];
#declare r=.03;
#declare d = 2; 
#declare a = 1.5;
 
#declare i=0;
#declare j=0;
#while(i<b)    
  #declare q[i]=
    < a*(cos(j)-2*sin(j)), a*(sin(j)+2*cos(j)), 2*d>;
  #declare p[i]=
    < a*(cos(j)+sin(j)), a*(sin(j)-cos(j)),-d>;
  #declare s[i]=
    < a*(cos(j)+2*sin(j)), a*(sin(j)-2*cos(j)), 2*d>;
  #declare o[i]=
    < a*(cos(j)-sin(j)), a*(sin(j)+cos(j)),-d>;
 cylinder { q[i], p[i], r pigment{Red}}          
 cylinder { s[i], o[i], r pigment{Blue}}          
 
 
  #declare j=j+(2*pi)/b;
  #declare i=i+1;                      
  
#end    
  

poly{2,<1/(a*a),0,0,0,1/(a*a),0,0,-1/(d*d),0,-1> 
	      pigment{Yellow*0.63  transmit 0.35}//} 
	      finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} 
	       	}

  // flat circular FINITE (no CSG) shape, center hole cutout is optional
/*
disc {
  <0, 0, d>  // center position
  z,         // normal vector
  sqrt(2*(a*a)), a - 0.01 
  texture {T_Silver_1A}
}     
disc {
  <0, 0, -d>  // center position
  z,         // normal vector
  sqrt(2*(a*a)), a-0.01 
  texture {T_Silver_1A}
}
*/

plane {
    z, -d
    texture {
      pigment {SkyBlue}   // Yellow
      /*pigment {
        checker
        color rgb<0.5, 0, 0>
        color rgb<0, 0.5, 0.5>
      }*/
      finish {
        diffuse 0.6
        ambient 0.2
        phong 1
        phong_size 100
        reflection 0.25
      }
    }
  }



