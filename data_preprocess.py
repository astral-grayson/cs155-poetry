import string
import re


def get_syllable(word):
    # gets number of syllables in given words

    syllable_dict = {}

    # compiling given syllable dictionary into dictionary
    with open("data/Syllable_dictionary.txt") as f:
        lines = f.readlines()
        for line in lines:
            lst = line.split()
            key = lst[0].translate(str.maketrans('', '', string.punctuation))

            # ignoring all ending syllable counts
            if len(lst) > 2 and ('E' in lst[len(lst) - 1]):
                syl = int(lst[len(lst) - 2])
            else:
                syl = int(lst[len(lst) - 1])

            syllable_dict[key] = syl

    return syllable_dict[word]



def parse_obs_list(text):
    # Convert text to dataset where text is list of lines

    obs_counter = 0
    obs = []
    obs_map = {}

    for line in text:
        obs_elem = []

        for word in line.split():
            word = re.sub(r'[^\w]', '', word).lower()
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1

            # Add the encoded word.
            obs_elem.append(obs_map[word])

        # Add the encoded sequence.
        obs.append(obs_elem)

    return obs, obs_map


def get_shakes_lines():
    # convert sonnets to list of lines

    shakes_lines = []
    with open("data/shakespeare.txt") as f:
        # Read in all lines
        lines = f.readlines()
        for line in lines:
            seq = line.strip()
            # remove punctuation
            seq = seq.translate(str.maketrans('', '', string.punctuation.replace("'", "").replace("-", "")))
            # make lowercase
            seq = seq.lower()
            # get rid of blank lines and numbers
            if len(seq) <= 3:
                continue
            #print(seq)
            shakes_lines.append(seq)
    f.close()

    return parse_obs_list(shakes_lines)


def get_shakes_sonnet():
    # convert sonnets to list of stanzas

    shakes_stanzas = []
    stanza = ""
    with open("data/shakespeare.txt") as f:
        # Read in all stanzas
        lines = f.readlines()
        for line in lines:
            seq = line.strip()
            # remove punctuation
            seq = seq.translate(str.maketrans('', '', string.punctuation.replace("'", "").replace("-", "")))
            # make lowercase
            seq = seq.lower()
            # get rid of blank lines and numbers
            if len(seq) <= 3 and len(stanza) == 0:
                continue
            # if this is the first time we've encountered a blank line, append the stanza
            elif len(seq) <= 3:
                shakes_stanzas.append(stanza)
                #print(stanza)
                stanza = ""
            else:
                stanza = stanza + seq + " "
    f.close()

    return parse_obs_list(shakes_stanzas)
