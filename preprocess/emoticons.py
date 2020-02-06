""" emoticon recognition via patterns.  tested on english-language twitter, but
probably works for other social media dialects. """

__author__ = "Brendan O'Connor (anyall.org, brenocon@gmail.com)"
__version__= "april 2009"

#from __future__ import print_function
import re,sys

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
#SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
#MULTITOK_SMILEY = mycompile(r' : [\)dp]')

NormalEyes = r'[:=]'
Wink = r'[;]'

NoseArea = r'(|o|O|-)'   ## rather tight precision, \S might be reasonable...

HappyMouths = r'[D\)\]]'
SurprisedMouths = r'[oO]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[d/\\]'  # remove forward slash if http://'s aren't cleaned

Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)
Surprised_RE = mycompile(NormalEyes + NoseArea + SurprisedMouths)
Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)
Other_RE = mycompile( '('+NormalEyes+'|'+Wink+')'  + NoseArea + OtherMouths )

Emoticon = (
    "("+NormalEyes+"|"+Wink+")" +
    NoseArea + 
    "("+Tongue+"|"+OtherMouths+"|"+SadMouths+"|"+HappyMouths+")"
)
Emoticon_RE = mycompile(Emoticon)

#Emoticon_RE = "|".join([Happy_RE,Sad_RE,Wink_RE,Tongue_RE,Other_RE])
#Emoticon_RE = mycompile(Emoticon_RE)

def analyze_tweet(text):
  h_= Happy_RE.search(text)
  s_= Sad_RE.search(text)
  #if h and s: return "BOTH_HS"
  #if h: return "HAPPY"
  #if s: return "SAD"
  #return "NA"

  # more complex & harder, so disabled for now
  w_= Wink_RE.search(text)
  t_= Tongue_RE.search(text)
  a_= Other_RE.search(text)
  sr_ = Surprised_RE.search(text)
  h,w,s,t,a, sr = [bool(x) for x in [h_,w_,s_,t_,a_, sr_]]
  #if sum([h,w,s,t,a, sr])>1: return "MULTIPLE SMILES"
  if sum([h,w,s,t,a, sr])>=1:
    if h: return (True, text[:h_.regs[0][0]] + "HAPPYSMILE" + text[h_.regs[0][1]:])
    if s: return (True, text[:s_.regs[0][0]] + "SADSMILE" + text[s_.regs[0][1]:])
    if w: return (True, text[:w_.regs[0][0]] + "WINKSMILE" + text[w_.regs[0][1]:])
    if sr: return (True, text[:sr_.regs[0][0]] + "SURPRISEDSMILE" + text[sr_.regs[0][1]:])
    if a: return (True, text[:a_.regs[0][0]] + "OTHERSMILE" + text[a_.regs[0][1]:])
    if t: return (True, text[:t_.regs[0][0]] + "TONGUESMILE" + text[t_.regs[0][1]:])
  return (False, text)

if __name__=='__main__':
  #for line in sys.stdin:
  line = "hello :O"
    #import sane_re
    #sane_re._S(line[:-1]).show_match(Emoticon_RE, numbers=False)
  print(analyze_tweet(line.strip()), line.strip(), sep="\t")
