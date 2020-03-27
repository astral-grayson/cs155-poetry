# HMM sonnet helper
# to avoid changing the HMM file

import numpy as np
import data_preprocess as data


def obs_map_reverser(obs_map):
    # reverses the obs map into words
    # from the original HMM helper class from set 6

    obs_map_r = {}

    for key in obs_map:
        obs_map_r[obs_map[key]] = key

    return obs_map_r


def sample_sonnet(hmm, obs_map):
    # make sonnet from trained HMM
    # similar to sample_sentence function from set 6

    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)
    sonnet = ''

    count = 0

    while count < 14:
        # Sample and convert sentence.
        n_words = np.random.choice(np.arange(1, 10))
        emission, states = hmm.generate_emission(n_words)
        sentence = [obs_map_r[i] for i in emission]
        syl_count = 0

        for word in sentence:
            syl_count += data.get_syllable(word)

        if syl_count == 10:
            sonnet += ' '.join(sentence).capitalize() + '\n'
            count += 1

    print(sonnet)
