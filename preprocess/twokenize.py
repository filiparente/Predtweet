# -*- coding: utf-8 -*-
""" tokenizer for tweets!  might be appropriate for other social media dialects too.
general philosophy is to throw as little out as possible.
development philosophy: every time you change a rule, do a diff of this
program's output on ~100k tweets.  if you iterate through many possible rules
and only accept the ones that seeem to result in good diffs, it's a sort of
statistical learning with in-the-loop human evaluation :)
"""

__author__="brendan o'connor (anyall.org)"

import re,sys
from . import emoticons
mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
def regex_or(*items):
  r = '|'.join(items)
  r = '(' + r + ')'
  return r
def pos_lookahead(r):
  return '(?=' + r + ')'
def neg_lookahead(r):
  return '(?!' + r + ')'
def optional(r):
  return '(%s)?' % r


PunctChars = r'''['“".?!,:;]'''
Punct = '%s+' % PunctChars
Entity = '&(amp|lt|gt|quot);'

# one-liner URL recognition:
#Url = r'''https?://\S+'''

# more complex version:
UrlStart1 = regex_or('https?://', r'www\.')
CommonTLDs = regex_or('com','co\\.uk','org','net','info','ca')
UrlStart2 = r'[a-z0-9\.-]+?' + r'\.' + CommonTLDs + pos_lookahead(r'[/ \W\b]')
UrlBody = r'[^ \t\r\n<>]*?'  # * not + for case of:  "go to bla.com." -- don't want period
UrlExtraCrapBeforeEnd = '%s+?' % regex_or(PunctChars, Entity)
UrlEnd = regex_or( r'\.\.+', r'[<>]', r'\s', '$')
Url = (r'\b' + 
    regex_or(UrlStart1, UrlStart2) + 
    UrlBody + 
    pos_lookahead( optional(UrlExtraCrapBeforeEnd) + UrlEnd))

Url_RE = re.compile("(%s)" % Url, re.U|re.I)

Timelike = r'\d+:\d+'
NumNum = r'\d+\.\d+'
NumberWithCommas = r'(\d+,)+?\d{3}' + pos_lookahead(regex_or('[^,]','$'))

Abbrevs1 = ['am','pm','us','usa','ie','eg']
def regexify_abbrev(a):
  chars = list(a)
  icase = ["[%s%s]" % (c,c.upper()) for c in chars]
  dotted = [r'%s\.' % x for x in icase]
  return "".join(dotted)
Abbrevs = [regexify_abbrev(a) for a in Abbrevs1]

BoundaryNotDot = regex_or(r'\s', '[“"?!,:;]', Entity)
aa1 = r'''([A-Za-z]\.){2,}''' + pos_lookahead(BoundaryNotDot)
aa2 = r'''([A-Za-z]\.){1,}[A-Za-z]''' + pos_lookahead(BoundaryNotDot)
ArbitraryAbbrev = regex_or(aa1,aa2)

assert '-' != '―'
Separators = regex_or('--+', '―')
Decorations = r' [  ♫   ]+ '.replace(' ','')

EmbeddedApostrophe = r"\S+'\S+"

ProtectThese = [
    #emoticons.Emoticon,
    Url,
    Entity,
    Timelike,
    #NumNum,
    #NumberWithCommas,
    #Punct,
    ArbitraryAbbrev,
    Separators,
    Decorations,
    #EmbeddedApostrophe,
]
Protect_RE = mycompile(regex_or(*ProtectThese))


class Tokenization(list):
  " list of tokens, plus extra info "
  def __init__(self):
    self.alignments = []
    self.text = ""
  def subset(self, tok_inds):
    new = Tokenization()
    new += [self[i] for i in tok_inds]
    new.alignments = [self.alignments[i] for i in tok_inds]
    new.text = self.text
    return new
  def assert_consistent(t):
    assert len(t) == len(t.alignments)
    assert [t.text[t.alignments[i] : (t.alignments[i]+len(t[i]))] for i in range(len(t))] == list(t)

def align(toks, orig):
  s_i = 0
  alignments = [None]*len(toks)
  for tok_i in range(len(toks)):
    while True:
      L = len(toks[tok_i])
      if orig[s_i:(s_i+L)] == toks[tok_i]:
        alignments[tok_i] = s_i
        s_i += L
        break
      s_i += 1
      if s_i >= len(orig): raise AlignmentFailed((orig,toks,alignments))
      #if orig[s_i] != ' ': raise AlignmentFailed("nonspace advance: %s" % ((s_i,orig),))
  if any(a is None for a in alignments): raise AlignmentFailed((orig,toks,alignments))

  return alignments

class AlignmentFailed(Exception): pass

def unicodify(s, encoding='utf8', *args):
  if isinstance(s,str): return s
  if isinstance(s,str): return s.decode(encoding, *args)
  return str(s)

def compound_word_split(compound_word):
    """
    Split a given compound word(string) and return list of words in given compound_word
    Ex: compound_word='pyTWEETCleaner' --> ['py', 'TWEET', 'Cleaner']
    """
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', compound_word)
    return [m.group(0) for m in matches]

def remove_non_ascii_chars(text):
    """
    return text after removing non-ascii characters i.e. characters with ascii value >= 128
    """
    return ''.join([w if ord(w) < 128 else ' ' for w in text])

def general_identifications(text):
    """
    replace @user_name by general @user for all user_names, because it adds no information
    """
    cleaned_text = ''
    for w in text.split(' '):
        if re.match(r'@[_A-Za-z0-9]+',w) and w[1:]!='bitcoin':
            cleaned_text = cleaned_text+ '@user '
        else:
            cleaned_text = cleaned_text + w + ' '
    return cleaned_text

    #return ' '.join([w for w in text.split(' ')  if '@' in w])

def replace_hyperlinks(text):
    """
    return text after removing hyperlinks
    """
    # Simply remove
    #return ' '.join([w for w in text.split(' ')  if not 'http' in w])

    # Replace by 'link'
    cleaned_text = ''
    for w in text.split(' '):
        if 'http' in w or 'www.' in w:
            cleaned_text = cleaned_text+ 'URL '
        else:
            cleaned_text = cleaned_text + w + ' '
    return cleaned_text
 
  
def get_cleaned_text(text, remove_retweets=True):
    """
    return cleaned text(string) for provided tweet text(string)
    """
    #cleaned_text = text.replace('\"','').replace('\'','').replace('-',' ')

    cleaned_text = text    
    cleaned_text =  remove_non_ascii_chars(cleaned_text)
    
    # retweet
    if re.match(r'RT @[_A-Za-z0-9]+:',text): # retweet
        if remove_retweets: return ''
        retweet_info = cleaned_text[:cleaned_text.index(':')+2] # 'RT @name: ' will be again added in the text after cleaning
        cleaned_text = cleaned_text[cleaned_text.index(':')+2:]
    else:
        retweet_info = ''
        
    cleaned_text = replace_hyperlinks(cleaned_text) 

    cleaned_text = general_identifications(cleaned_text)

    while(1):
      (has_emoticon, cleaned_text) = emoticons.analyze_tweet(cleaned_text.strip())
      if not has_emoticon:
        break
    
    cleaned_text = cleaned_text.replace('# ','HASHTAG ').replace('@ ','AT ') # to avoid being removed while removing punctuations
    
    #tokens = [w.translate(self.punc_table) for w in word_tokenize(cleaned_text)] # remove punctuations and tokenize
    #tokens = [w for w in tokens if not w.lower() in self.stop_words and len(w)>1] # remove stopwords and single length words
    #cleaned_text = ' '.join(tokens)
    
    #cleaned_text = cleaned_text.replace('HASHTAGSYMBOL','#').replace('ATSYMBOL','@')
    cleaned_text = retweet_info + cleaned_text
    
    return cleaned_text

def tokenize(tweet):
  text = unicodify(tweet)
  text = squeeze_whitespace(text)
  text = get_cleaned_text(text)
  t = Tokenization()
  t += simple_tokenize(text)
  t.text = text
  t.alignments = align(t, text)
  return t

def simple_tokenize(text):
  s = text
  s = edge_punct_munge(s)

  # strict alternating ordering through the string.  first and last are goods.
  # good bad good bad good bad good
  goods = []
  bads = []
  i = 0
  if Protect_RE.search(s):
    for m in Protect_RE.finditer(s):
      goods.append( (i,m.start()) )
      bads.append(m.span())
      i = m.end()
    goods.append( (m.end(), len(s)) )
  else:
    goods = [ (0, len(s)) ]
  assert len(bads)+1 == len(goods)

  goods = [s[i:j] for i,j in goods]
  bads  = [s[i:j] for i,j in bads]
  #print goods
  #print bads
  goods = [unprotected_tokenize(x) for x in goods]
  res = []
  for i in range(len(bads)):
    res += goods[i]
    #res.append(bads[i])
  res += goods[-1]

  res = post_process(res)
  return res

AposS = mycompile(r"(\S+)('s)$")

def post_process(pre_toks):
  # hacky: further splitting of certain tokens
  post_toks = []
  for tok in pre_toks:
    m = AposS.search(tok)
    if m:
      post_toks += m.groups()
    else:
      post_toks.append( tok )
  return post_toks

WS_RE = mycompile(r'\s+')
def squeeze_whitespace(s):
  new_string = WS_RE.sub(" ",s)
  return new_string.strip()

# fun: copy and paste outta http://en.wikipedia.org/wiki/Smart_quotes
EdgePunct      = r"""[  ' " “ ” ‘ ’ < > « » { } ( \) [ \]  ]""".replace(' ','')
#NotEdgePunct = r"""[^'"([\)\]]"""  # alignment failures?
NotEdgePunct = r"""[a-zA-Z0-9]"""
EdgePunctLeft  = r"""(\s|^)(%s+)(%s)""" % (EdgePunct, NotEdgePunct)
EdgePunctRight =   r"""(%s)(%s+)(\s|$)""" % (NotEdgePunct, EdgePunct)
EdgePunctLeft_RE = mycompile(EdgePunctLeft)
EdgePunctRight_RE= mycompile(EdgePunctRight)

def edge_punct_munge(s):
  s = EdgePunctLeft_RE.sub( r"\1\2 \3", s)
  s = EdgePunctRight_RE.sub(r"\1 \2\3", s)
  return s


def unprotected_tokenize(s):
  return s.split()

if __name__=='__main__':
  for line in sys.stdin:
    print(u" ".join(tokenize(line[:-1])).encode('utf-8'))
    #print "CUR\t" + " ".join(tokenize(line[:-1]))
    #print "WS\t" + " ".join(line[:-1].split())
    #print ansi.color(line.strip(),'red')
    #print ansi.color(" ".join(tokenize(line.strip())),'blue','bold')
