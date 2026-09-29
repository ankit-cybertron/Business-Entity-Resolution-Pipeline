import re
import unicodedata

# CELL 3 — TEXT NORMALIZATION (English + Indic + French)
_SCRIPTS = ("DEVANAGARI","BENGALI","GURMUKHI","GUJARATI","ORIYA","ODIA",
            "TAMIL","TELUGU","KANNADA","MALAYALAM")
_LETTER = {"KA":"ka","KHA":"kha","GA":"ga","GHA":"gha","NGA":"nga","CA":"cha","CHA":"chha",
           "JA":"ja","JHA":"jha","NYA":"nya","TTA":"ta","TTHA":"tha","DDA":"da","DDHA":"dha",
           "NNA":"na","TA":"ta","THA":"tha","DA":"da","DHA":"dha","NA":"na","PA":"pa",
           "PHA":"pha","BA":"ba","BHA":"bha","MA":"ma","YA":"ya","RA":"ra","LA":"la",
           "LLA":"la","VA":"va","SHA":"sha","SSA":"sha","SA":"sa","HA":"ha","QA":"qa",
           "KHHA":"kha","GHHA":"gha","ZA":"za","DDDA":"ra","RHA":"rha","FA":"fa",
           "YYA":"ya","RRA":"ra","NGA2":"nga","NNN":"na","NNNA":"na","NNNN":"na",
           "LL":"la","LLL":"la","RR":"ra","NN":"na","N":"na","TH":"tha","SH":"sha","SS":"sha"}
_VOWEL = {"A":"a","AA":"aa","I":"i","II":"ee","U":"u","UU":"oo","VOCALIC R":"ri",
          "VOCALIC RR":"ri","VOCALIC L":"li","VOCALIC LL":"li","E":"e","EE":"ee",
          "AI":"ai","O":"o","OO":"oo","AU":"au","SHORT E":"e","SHORT O":"o",
          "CANDRA E":"e","CANDRA O":"o"}
_CONSONANTS, _VOWELS, _MATRAS = {}, {}, {}
_VIRAMA, _ANUSVARA, _VISARGA, _DIGITS = set(), set(), set(), {}
for _cp in range(0x0900, 0x0D80):
    try: _name = unicodedata.name(chr(_cp))
    except ValueError: continue
    if not any(_name.startswith(s) for s in _SCRIPTS): continue
    _rest = _name.split(" ",1)[1] if " " in _name else ""
    if _rest.startswith("LETTER "):
        _k = _rest[7:]
        if _k in _LETTER: _CONSONANTS[_cp] = _LETTER[_k]
        elif _k in _VOWEL: _VOWELS[_cp] = _VOWEL[_k]
    elif _rest.startswith("VOWEL SIGN "):
        _k = _rest[11:]
        if _k in _VOWEL: _MATRAS[_cp] = _VOWEL[_k]
    elif _rest in ("SIGN VIRAMA","VIRAMA"): _VIRAMA.add(_cp)
    elif _rest.startswith("SIGN ANUSVARA") or _rest.startswith("SIGN CANDRABINDU"): _ANUSVARA.add(_cp)
    elif _rest.startswith("SIGN VISARGA"): _VISARGA.add(_cp)
    elif _rest.startswith("DIGIT "):
        _d = _rest[6:]
        if _d.isdigit(): _DIGITS[_cp] = _d

_INDIC_RE = re.compile("[ऀ-ൿ]"); _ZW = "​‌‍﻿"
def _translit(s):
    out=[]; i,n=0,len(s)
    while i<n:
        cp=ord(s[i])
        if cp in _CONSONANTS:
            out.append(_CONSONANTS[cp]); i+=1
            while i<n:
                ncp=ord(s[i])
                if s[i] in _ZW: i+=1
                elif ncp in _VIRAMA: out[-1]=out[-1][:-1]; i+=1; break
                elif ncp in _MATRAS: out[-1]=out[-1][:-1]+_MATRAS[ncp]; i+=1; break
                elif ncp in _ANUSVARA: out.append("n"); i+=1
                elif ncp in _VISARGA: out.append("h"); i+=1
                else: break
        elif cp in _VOWELS: out.append(_VOWELS[cp]); i+=1
        elif cp in _ANUSVARA: out.append("n"); i+=1
        elif cp in _VISARGA: out.append("h"); i+=1
        elif cp in _DIGITS: out.append(_DIGITS[cp]); i+=1
        else: out.append(" " if not s[i].isalnum() else s[i]); i+=1
    return "".join(out)

LEGAL = {"inc","incorporated","llc","lcsw","lp","llp","ltd","limited","pvt","private","pte",
         "plc","corp","corporation","company","gmbh","sa","srl","bv","nv","ag","sas","sarl",
         "pty","opc","pc","pa","lllp","pllc","psc","cic","ulc","sbc","eurl","sasu","snc",
         "scs","sca","sci","scp","eirl","earl","selarl","sem","scop","scic","gie"}
WEB = {"www","com","net","org","info","biz","io","http","https","html","php","aspx"}
TLD_TAIL = {"in","us","co","uk","au","ca","fr"}
ADDR_ABBREV = {"rd":"road","str":"street","ave":"avenue","av":"avenue","blvd":"boulevard",
               "dr":"drive","ln":"lane","hwy":"highway","pkwy":"parkway","ct":"court","pl":"place",
               "sq":"square","hno":"house","hn":"house","no":"number","num":"number","opp":"opposite",
               "nr":"near","bldg":"building","bldgs":"building","flr":"floor","fl":"floor",
               "apt":"apartment","dept":"department","ste":"suite","rm":"room","blk":"block",
               "sect":"sector","dist":"district","distt":"district","po":"post","bd":"boulevard",
               "bld":"boulevard","che":"chemin","chem":"chemin","imp":"impasse","rle":"ruelle",
               "rte":"route","all":"allee","allee":"allee","qu":"quai","quai":"quai",
               "fbg":"faubourg","crs":"cours","pas":"passage","rpt":"rond point"}
STOP = {"the","and","of","a","an","for","at","on","in","to","st","str","n","de","du","des",
        "la","le","les","l","d","au","aux","et","en","sur","sous","un","une"}
_PUNCT = re.compile(r"[^0-9a-zऀ-ൿ]+"); _WS = re.compile(r"\s+")
_LATIN_ACCENT = re.compile(r"[̀-ͯ]"); _NUMERIC = re.compile(r"\d+")

def fold(s):
    if not s: return ""
    s = (s.replace("Œ","OE").replace("œ","oe").replace("Æ","AE").replace("æ","ae")
         .replace("N°"," ").replace("Nº"," ").replace("n°"," ").replace("nº"," "))
    s = unicodedata.normalize("NFKD", s)
    s = _LATIN_ACCENT.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    return s.lower()

def normalize(s):
    if not s: return ""
    s = fold(s)
    if _INDIC_RE.search(s): s = _translit(s)
    s = s.replace("&"," and ")
    return _WS.sub(" ", _PUNCT.sub(" ", s)).strip()

def _dedup(toks):
    out=[t for i,t in enumerate(toks) if i==0 or t!=toks[i-1]]
    if len(out)>1:
        seen={}
        for t in out: seen[t]=seen.get(t,0)+1
        once=[t for t in out if seen[t]==1]
        out=once or out
    return out

def name_core(s):
    toks = normalize(s).split()
    if len(toks)>1 and toks[-1] in TLD_TAIL: toks = toks[:-1]
    toks = [t for t in toks if t not in LEGAL and t not in WEB]
    return " ".join(_dedup(toks))

def addr_core(s):
    out=[]
    for t in normalize(s).split():
        t = ADDR_ABBREV.get(t, t)
        if t in LEGAL or t in STOP: continue
        for p in t.split():
            if p not in LEGAL and p not in STOP: out.append(p)
    return " ".join(out)

_VOWEL_CHARS = str.maketrans("","","aeiou")
def skeleton(s): return normalize(s).replace(" ","").translate(_VOWEL_CHARS)

_PH_DIGRAPHS = [("chh","C"),("ch","C"),("sh","S"),("ph","F"),("th","T"),("kh","K"),("gh","K"),
                ("bh","P"),("dh","T"),("jh","C"),("ng","N"),("ck","K"),("qu","K"),
                ("ss","S"),("zz","S"),("ce","S"),("ci","S"),("cy","S")]
_PH_CLASS = {"C":"S","S":"S","F":"P","T":"T","K":"K","P":"P","N":"N","c":"K","k":"K","g":"K",
             "q":"K","p":"P","b":"P","f":"P","v":"P","w":"P","t":"T","d":"T","s":"S","z":"S",
             "x":"S","j":"S","m":"M","n":"N","r":"R","l":"L","h":"H","y":"Y"}
_PH_KEEP=re.compile(r"[^a-z0-9CSFTKPNR]"); _PH_REPEAT=re.compile(r"(.)\1+")
_PH_VOWELS=str.maketrans("","","aeiou")
def phonetic(s):
    t = normalize(s).replace(" ","")
    for a,b in _PH_DIGRAPHS: t = t.replace(a,b)
    t = t.translate(_PH_VOWELS)
    t = _PH_KEEP.sub("", t)
    t = "".join(_PH_CLASS.get(c,c) for c in t)
    return _PH_REPEAT.sub(r"\1", t)

def num_tokens(s): return set(_NUMERIC.findall(fold(s)))

# Self-check
assert name_core("Cœur Défense") == "coeur defense"
assert name_core("Dupont SAS") == "dupont"
assert addr_core("12 Bd de la Paix, 75002 Paris") == "12 boulevard paix 75002 paris"
assert skeleton("Nétwork") == "ntwrk"


def normalize_record(row):
    eid, country, name, addr = row
    name = "" if name is None else str(name)
    addr = "" if addr is None else str(addr)
    return (
        eid, country, name, addr,
        name_core(name), phonetic(name), skeleton(name), addr_core(addr)
    )
