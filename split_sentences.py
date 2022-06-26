"""
Code copied from https://gist.github.com/quynguyen3490/04cc1178efd0ae06bff7b59627626f9b
Author: Quy Nguyen
"""
import re

# tất cả ký tự viết hoa tiếng Việt
UPPERUNICODE = [u'\u00C1', u'\u00C0', u'\u00C2', u'\u0102', u'\u00C3', u'\u1EA4', u'\u1EA6', u'\u1EAE', u'\u1EB0', u'\u1EAA', u'\u1EB4', u'\u1EA2', u'\u1EA8', u'\u1EB2', u'\u1EA0', u'\u1EAC', u'\u1EB6', u'\u0110', u'\u00C9', u'\u00C8', u'\u00CA', u'\u1EBC', u'\u1EBE', u'\u1EC0', u'\u1EC4', u'\u1EBA', u'\u1EC2', u'\u1EB8', u'\u1EC6', u'\u00CD', u'\u00CC', u'\u0128', u'\u1EC8',
                u'\u1ECA', u'\u00D3', u'\u00D2', u'\u00D4', u'\u00D5', u'\u1ED0', u'\u1ED2', u'\u1ED6', u'\u1ECE', u'\u01A0', u'\u1ED4', u'\u1ECC', u'\u1EDA', u'\u1EDC', u'\u1EE0', u'\u1ED8', u'\u1EDE', u'\u1EE2', u'\u00DA', u'\u00D9', u'\u0168', u'\u1EE6', u'\u01AF', u'\u1EE4', u'\u1EE8', u'\u1EEA', u'\u1EEE', u'\u1EEC', u'\u1EF0', u'\u00DD', u'\u1EF2', u'\u1EF8', u'\u1EF6', u'\u1EF4']
# tất cả ký tự viết thường tiếng Việt
LOWERUNICODE = [u'\u00E1', u'\u00E0', u'\u00E2', u'\u0103', u'\u00E3', u'\u1EA5', u'\u1EA7', u'\u1EAF', u'\u1EB1', u'\u1EAB', u'\u1EB5', u'\u1EA3', u'\u1EA9', u'\u1EB3', u'\u1EA1', u'\u1EAD', u'\u1EB7', u'\u0111', u'\u00E9', u'\u00E8', u'\u00EA', u'\u1EBD', u'\u1EBF', u'\u1EC1', u'\u1EC5', u'\u1EBB', u'\u1EC3', u'\u1EB9', u'\u1EC7', u'\u00ED', u'\u00EC', u'\u0129', u'\u1EC9',
                u'\u1ECB', u'\u00F3', u'\u00F2', u'\u00F4', u'\u00F5', u'\u1ED1', u'\u1ED3', u'\u1ED7', u'\u1ECF', u'\u01A1', u'\u1ED5', u'\u1ECD', u'\u1EDB', u'\u1EDD', u'\u1EE1', u'\u1ED9', u'\u1EDF', u'\u1EE3', u'\u00FA', u'\u00F9', u'\u0169', u'\u1EE7', u'\u01B0', u'\u1EE5', u'\u1EE9', u'\u1EEB', u'\u1EEF', u'\u1EED', u'\u1EF1', u'\u00FD', u'\u1EF3', u'\u1EF9', u'\u1EF7', u'\u1EF5']
# ký tự được xem là kết thúc câu
BOUDARYSYMBOL = ['.', '!', '?', u'\u2026', u'…']
# ký tự trắng
WHITE_SPACE = ' '
# ký tự dấu ba chấm
ELLIPSIS = u'\u2026'

# kiểm tra ký tự kết thúc câu


def isBoundarySymbol(char):
    for c in BOUDARYSYMBOL:
        if c == char:
            return 1
    return 0

# kiểm tra ký tự viết hoa tiếng Việt


def isUniUpper(char):
    for c in UPPERUNICODE:
        if c == char:
            return 1
    return 0
# kiếm tra ký tự viết thường tiếng Việt


def isUniLower(char):
    for c in LOWERUNICODE:
        if c == char:
            return 1
    return 0

# kiểm tra có phải kết thúc 1 câu hay ko


def isBoudary(i, text):
    # kiểm tra hết text
    if i == len(text)-1:
        return 1

    # nếu i+1 là ký tự trắng thì trả về 1
    if re.match(r'\s', text[i+1]):
        return 1

    # nếu i-1 là số và i+1 là số thì trả về 0
    if re.match(r'[0-9]', text[i-1]) and re.match(r'[0-9]', text[i+1]):
        return 0

    # nếu i-1 là số và i+1 không phải là số thì trả về 1
    if re.match(r'[0-9]', text[i-1]) and re.match(r'[^0-9]', text[i+1]):
        return 1

    # nếu i-1 là ký tự thường và i+1 là ký tự hoa thì trả về 1
    if (re.match(r'[a-z\)\"\”]', text[i-1]) or isUniLower(text[i-1])) and (re.match(r'[A-Z\(\"\“)]', text[i+1]) or isUniUpper(text[i+1])):
        return 1

    # nếu i-1 là ký tự hoa và i+1 là ký tự hoa thì trả về 0
    if (re.match(r'[A-Z]', text[i-1]) or isUniUpper(text[i-1])) and (re.match(r'[A-Z]', text[i+1]) or isUniUpper(text[i+1])):
        return 0

    return 0

# cắt câu


def SplitSentence(text):
    text = text.strip()
    text = text.replace('\t', u'. ')
    text = text.replace('\v', u'. ')
    text = text.replace('\r', u'. ')
    text = text.replace('\r\n', u'. ')
    text = text.replace(u'...', u'. ')
    text = text.replace(u'..', u'. ')
    text = text.replace(u'…', u'. ')

    sents = []

    # ký tự i bắt đầu =0
    i = 0
    # ký tự bắt đầu câu =0
    begin = 0

    # lặp từng ký tự trong text
    while i < len(text):
        for sym in BOUDARYSYMBOL:
            # nếu gặp ký tự kết thúc
            if sym == text[i]:
                # kiểm tra đó có phải vị trí kết thúc câu hay ko
                if isBoudary(i, text):
                    # nếu là kết thúc câu thì cắt câu đó ra và tiếp tục
                    sents.append(text[begin:i].strip())
                    begin = i+1
        i += 1
    return sents
