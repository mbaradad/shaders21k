def get_examples_per_mode():
  return [(('classic', False, False, False),"""precision highp float;
uniform vec2 resolution;
uniform vec2 mouse;
uniform float time;
uniform sampler2D backbuffer;
void main(){vec2 r=resolution,p=(gl_FragCoord.xy*2.-r)/min(r.x,r.y)-mouse;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(time*.2)*.4);}gl_FragColor=vec4(p.xxy,1);}
"""),
   (('geek', False, False, False),"""precision highp float;
uniform vec2 r;
uniform vec2 m;
uniform float t;
uniform sampler2D b;
void main(){vec2 p=(gl_FragCoord.xy*2.-r)/min(r.x,r.y)-m;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(t*.2)*.4);}gl_FragColor=vec4(p.xxy,1);}
"""),
   (('geeker', False, False, False),"""void main(){vec2 p=(gl_FragCoord.xy*2.-r)/min(r.x,r.y)-m;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(t*.2)*.4);}gl_FragColor=vec4(p.xxy,1);}
"""),
   (('geekest', False, False, False),"""vec2 p=(FC.xy*2.-r)/min(r.x,r.y)-m;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(t*.2)*.4);}gl_FragColor=vec4(p.xxy,1);
"""),

   (('classic', True, False, False),"""precision highp float;
uniform vec2 resolution;
uniform vec2 mouse;
uniform float time;
uniform sampler2D backbuffer;
out vec4 outColor;
void main(){vec2 r=resolution,p=(gl_FragCoord.xy*2.-r)/min(r.x,r.y)-mouse;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(time*.2)*.4);}outColor=vec4(p.xxy,1);}
"""),
   (('geek', True, False, False),"""precision highp float;
uniform vec2 r;
uniform vec2 m;
uniform float t;
uniform sampler2D b;
out vec4 o;
void main(){vec2 p=(gl_FragCoord.xy*2.-r)/min(r.x,r.y)-m;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(t*.2)*.4);}o=vec4(p.xxy,1);}
"""),
   (('geeker', True, False, False),"""void main(){vec2 p=(gl_FragCoord.xy*2.-r)/min(r.x,r.y)-m;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(t*.2)*.4);}o=vec4(p.xxy,1);}
"""),
   (('geekest', True, False, False),"""vec2 p=(FC.xy*2.-r)/min(r.x,r.y)-m;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(t*.2)*.4);}o=vec4(p.xxy,1);
"""),

   (('classic', False, True, False),"""precision highp float;
uniform vec2 resolution;
uniform vec2 mouse;
uniform float time;
uniform sampler2D backbuffer0;
uniform sampler2D backbuffer
layout (location = 0) out vec4 outColor0;
layout (location = 1) out vec4 outColor1;
void main(){vec2 r=resolution,p=(gl_FragCoord.xy*2.-r)/min(r.x,r.y)-mouse;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(time*.2)*.4);}outColor0=vec4(p.xxy,1);outColor1=outColor0;}
"""),
   (('geek', False, True, False),"""precision highp float;
uniform vec2 r;
uniform vec2 m;
uniform float t;
uniform sampler2D b;
layout (location = 0) out vec4 o0;
layout (location = 1) out vec4 o1;
void main(){vec2 p=(gl_FragCoord.xy*2.-r)/min(r.x,r.y)-m;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(t*.2)*.4);}o0=vec4(p.xxy,1);o1=o0;}
"""),
   (('geeker', False, True, False),"""void main(){vec2 p=(gl_FragCoord.xy*2.-r)/min(r.x,r.y)-m;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(t*.2)*.4);}o0=vec4(p.xxy,1);o1=o0;}
"""),
   (('geekest', False, True, False),"""vec2 p=(FC.xy*2.-r)/min(r.x,r.y)-m;for(int i=0;i<8;++i){p.xy=abs(p)/abs(dot(p,p))-vec2(.9+cos(t*.2)*.4);}o0=vec4(p.xxy,1);o1=o0;
"""),
  ]
