# Copyright 2017 Bo Shao. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
from collections import defaultdict

AUG0_FOLDER = "Augment0"
AUG1_FOLDER = "Augment1"
AUG2_FOLDER = "Augment2"

VOCAB_FILE = "vocab.txt"
CORNELL_DATA_FILE = "cornell_cleaned_new.txt"
REDDIT_DATA_FILE = "reddit_cleaned_part.txt"
OPENSUBTITLES_DATA_FILE = "opensubtitles_full_cleaned.txt"
BAD_WORDS_FILE = "bad-words.txt"
EXCLUDED_FILE = "excluded.txt"
VOCAB_MAX_SIZE = 20000


def load_bad_words(corpus_dir):
    bad = set()
    with open(os.path.join(corpus_dir, BAD_WORDS_FILE), 'r') as f:
        for line in f:
            word = line.strip().lower();
            bad.add(word)
    return bad


def generate_vocab_file(corpus_dir):
    """
    Generate the vocab.txt file for the training and prediction/inference.
    Manually remove the empty bottom line in the generated file.
    """
    vocab_list = []

    # Special tokens, with IDs: 0, 1, 2
    for t in ['_eos_', '_bos_', '_unk_']:
        vocab_list.append(t)

    # The word following this punctuation should be capitalized in the prediction output.
    for t in ['.', '!', '?']:
        vocab_list.append(t)

    # The word following this punctuation should not precede with a space in the prediction output.
    for t in ['(', '[', '{', '``', '$']:
        vocab_list.append(t)

    vocab_dict = defaultdict(lambda: 0)
    bad_words = load_bad_words(corpus_dir);

    for fd in range(2, -1, -1):
        if fd == 0:
            file_dir = os.path.join(corpus_dir, AUG0_FOLDER)
        elif fd == 1:
            file_dir = os.path.join(corpus_dir, AUG1_FOLDER)
        else:
            file_dir = os.path.join(corpus_dir, AUG2_FOLDER)

        for data_file in sorted(os.listdir(file_dir)):
            full_path_name = os.path.join(file_dir, data_file)
            if os.path.isfile(full_path_name) and data_file.lower().endswith('.txt'):
                if fd == 0 and (data_file == CORNELL_DATA_FILE or data_file == REDDIT_DATA_FILE or data_file == OPENSUBTITLES_DATA_FILE):
                    continue  # Will be processed below
                with open(full_path_name, 'r') as f:
                    for line in f:
                        l = line.strip()
                        if not l:
                            continue
                        if l.startswith("Q:") or l.startswith("A:"):
                            tokens = l[2:].strip().split(' ')
                            for token in tokens:
                                if len(token) and token != ' ':
                                    t = token.lower()
                                    if t not in vocab_list and t not in bad_words:
                                        vocab_list.append(t)

    #print("Vocab size after all base data files scanned: {}".format(len(vocab_list)))
    vocab_set = set(vocab_list)

    temp_dict = {}  # A temp dict
    cornell_file = os.path.join(corpus_dir, AUG0_FOLDER, CORNELL_DATA_FILE)
    if os.path.isfile(cornell_file):
        with open(cornell_file, 'r') as f1:
            for line in f1:
                ln = line.strip()
                if not ln:
                    continue
                if ln.startswith("Q:") or ln.startswith("A:"):
                    tokens = ln[2:].strip().split(' ')
                    for token in tokens:
                        if len(token) and token != ' ':
                            t = token.lower()
                            if t not in vocab_set and t not in bad_words:
                                vocab_dict[t] += 1

    print("Vocab size after cornell data file scanned: {}".format(len(vocab_list)))

    reddit_file = os.path.join(corpus_dir, AUG0_FOLDER, REDDIT_DATA_FILE)
    if os.path.isfile(reddit_file):
        with open(reddit_file, 'r') as f2:
            line_cnt = 0
            for line in f2:
                line_cnt += 1
                if line_cnt % 200000 == 0:
                    print("{:,} lines of reddit data file scanned.".format(line_cnt))
                ln = line.strip()
                if not ln:
                    continue
                if ln.startswith("Q:") or ln.startswith("A:"):
                    tokens = ln[2:].strip().split(' ')
                    for token in tokens:
                        if len(token) and token != ' ':
                            t = token.lower()
                            if t not in vocab_set and t not in bad_words:
                                vocab_dict[t] += 1

    opensubtitles_file = os.path.join(corpus_dir, AUG0_FOLDER, OPENSUBTITLES_DATA_FILE)
    if os.path.isfile(opensubtitles_file):
        with open(opensubtitles_file, 'r') as f2:
            line_cnt = 0
            for line in f2:
                line_cnt += 1
                if line_cnt % 200000 == 0:
                    print("{:,} lines of opensubtitles data file scanned.".format(line_cnt))
                ln = line.strip()
                if not ln:
                    continue
                if ln.startswith("Q:") or ln.startswith("A:"):
                    tokens = ln[2:].strip().split(' ')
                    for token in tokens:
                        if len(token) and token != ' ':
                            t = token.lower()
                            if t not in vocab_set and t not in bad_words:
                                vocab_dict[t] += 1

    more = VOCAB_MAX_SIZE - len(vocab_list)
    if more > 0:
        limited = sorted(vocab_dict.items(), key=lambda x: -x[1])[:more]
    vocab_list.extend(map(lambda x: x[0], limited))

    with open(os.path.join(corpus_dir, VOCAB_FILE), 'w') as f_voc:
        for v in vocab_list:
            f_voc.write("{}\n".format(v))

    print("The final vocab file generated. Vocab size: {}".format(len(vocab_list)))

    with open(os.path.join(corpus_dir, EXCLUDED_FILE), 'w') as f_excluded:
        for k, _ in temp_dict.items():
            if k not in vocab_list:
                f_excluded.write("{}\n".format(k))

if __name__ == "__main__":
    corp_dir = os.path.join('.', 'Data', 'Corpus')
    generate_vocab_file(corp_dir)
