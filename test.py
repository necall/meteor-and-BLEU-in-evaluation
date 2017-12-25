#!/usr/bin/env python
import argparse  # optparse is deprecated
from itertools import islice  # slicing for iterators
import math
import nltk


def word_matches(h, ref):
    # print h,'+++',ref
    # print sum(1 for w in h if w in ref)
    return sum(1 for w in h if w in ref)


def word_bleu(h, ref):
    totalprecision = 1
    for i in range(1, 3):
        if i > len(h) or i > len(ref):
            return 0
        hsubtotal = getlist(h, i)
        refsubtotal = getlist(ref, i)

        # print hsubtotal,refsubtotal,hpos,refpos
        count = 0
        for hh in hsubtotal:
            if hh in refsubtotal:
                count += 1
        precision = float(count) / len(hsubtotal)

        # pcount=0
        # for hp in hpos:
        #     if hp in refpos:
        #         pcount+=1
        # pprecision=float(pcount)/len(hpos)
        totalprecision *= precision
        # print precision

    bp = math.pow(math.e, (1 - float(len(ref)) / len(h)))
    geometric_mean = totalprecision ** (1 / float(2))

    bleu = bp * geometric_mean
    return bleu


def getlist(l, i):
    subtotal = []
    # postotal=[]
    # l=list(l)
    # l=nltk.pos_tag(ll)
    for index in range(len(l)):
        ltemp = []
        ltemp.append(l[index])

        if i == 1:
            pass
        else:
            if index + i - 1 < len(l):
                for ii in range(1, i):
                    ltemp.append(l[index + ii])
            else:
                continue
        subtotal.append(ltemp)
    return subtotal


def word_meteor(h, ref):
    # print h,ref
    # texth=nltk.word_tokenize(h)
    hh = nltk.pos_tag(h)
    # print hh
    # textref=nltk.word_tokenize(ref)
    reff = nltk.pos_tag(ref)

    wordcount = sum(1 for w in hh if w in reff)

    P = float(wordcount) / len(h)
    R = float(wordcount) / len(ref)
    # tune index
    if wordcount == 0:
        return 0
    index = 0.8
    # print P,'++',R
    meteor = P * R / ((1 - index) * R + index * P)
    return meteor


def word_net(h, ref):
    wordcount = 0
    synlist = []
    for hh in h:
        # print hh
        hhafter = []
        try:
            hhafter = wordnet.synsets(hh)
        except UnicodeDecodeError:
            pass

        for syn in hhafter:
            for l in syn.lemmas():
                if l.name() not in synlist:
                    synlist.append(l.name())
    setsyn = set(synlist)
    for rr in ref:
        if rr in setsyn:
            wordcount += 1
    return wordcount


def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
                        help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
                        help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()

    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        rset = set(ref)
        # h1_match = word_matches(h1, rset)
        # h2_match = word_matches(h2, rset)
        # print(1 if h1_match > h2_match else # \begin{cases}
        #         (0 if h1_match == h2_match
        #             else -1)) # \end{cases}
        h1_meteor = word_meteor(h1, rset)
        h2_meteor = word_meteor(h2, rset)
        # print(1 if h1_meteor > h2_meteor else  # \begin{cases}
        #       (0 if h1_meteor == h2_meteor
        #        else -1))  # \end{cases}
        h1_bleu = word_bleu(h1, ref)
        h2_bleu = word_bleu(h2, ref)
        h1_score = 0.99 * h1_bleu + 0.01 * h1_meteor
        h2_score = 0.99 * h2_bleu + 0.01 * h2_meteor
        print(1 if h1_score > h2_score else  # \begin{cases}
              (0 if h1_score == h2_score
               else -1))  # \end{cases}


# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
