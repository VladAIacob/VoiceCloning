###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import re
import os
import argparse
import json
import random
import numpy as np
import torch
import torch.utils.data
from scipy.io.wavfile import read
from scipy.stats import betabinom
from audio_processing import TacotronSTFT
from text import text_to_sequence, cmudict, _clean_text, get_arpabet


def beta_binomial_prior_distribution(phoneme_count, mel_count):
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = i, M+1-i
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def load_filepaths_and_text(filelist, split="|"):
    if isinstance(filelist, str):
        with open(filelist, encoding='utf-8') as f:
            filepaths_and_text = [line.strip().split(split) for line in f]
    else:
        filepaths_and_text = filelist
    return filepaths_and_text


def load_wav_to_torch(full_path):
    """ Loads wavdata into torch array """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class Data(torch.utils.data.Dataset):
    def __init__(self, filelist_path, filter_length, hop_length, win_length,
                 sampling_rate, mel_fmin, mel_fmax, max_wav_value, p_arpabet,
                 cmudict_path, text_cleaners, speaker_ids=None,
                 use_attn_prior=False, attn_prior_threshold=1e-4,
                 randomize=True, keep_ambiguous=False, seed=1234):
        self.max_wav_value = max_wav_value
        self.audiopaths_and_text = load_filepaths_and_text(filelist_path)
        self.use_attn_prior = use_attn_prior
        self.attn_prior_threshold = attn_prior_threshold
        self.keep_ambiguous=keep_ambiguous

        if speaker_ids is None or speaker_ids == '':
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)
        else:
            self.speaker_ids = speaker_ids

        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.sampling_rate = sampling_rate
        self.text_cleaners = text_cleaners
        self.p_arpabet = p_arpabet
        self.cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=keep_ambiguous)
        '''if speaker_ids is None:
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)
        else:
            self.speaker_ids = speaker_ids
		  '''
        random.seed(seed)
        if randomize:
            random.shuffle(self.audiopaths_and_text)

    def compute_attention_prior(self, audiopath, mel_length, text_length):
        attn_prior = beta_binomial_prior_distribution(text_length, mel_length)

        if self.attn_prior_threshold > 0:
            attn_prior = attn_prior.masked_fill(
                attn_prior < self.attn_prior_threshold, 0.0)

        return attn_prior

    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        print("Number of speakers :", len(d))
        return d

    def get_mel(self, audio):
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def get_speaker_id(self, speaker_id):
        return torch.LongTensor([self.speaker_ids[int(speaker_id)]])

    def get_text(self, text):
        text = _clean_text(text, self.text_cleaners)
        words = re.findall(r'\S*\{.*?\}\S*|\S+', text)
        text = ' '.join([get_arpabet(word, self.cmudict)
                         if random.random() < self.p_arpabet else word
                         for word in words])
        text_norm = torch.LongTensor(text_to_sequence(text))
        return text_norm
        
    def get_embeds(self, embeds):
        return torch.from_numpy(np.array(embeds))

    def __getitem__(self, index):
        # Read audio and text
        audiopath, text, speaker_id = self.audiopaths_and_text[index]
        audio, sampling_rate = load_wav_to_torch(audiopath)
        embeds = np.load(audiopath.replace(".wav", ".npy"))
        '''if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR, in {}".format(
                sampling_rate, self.sampling_rate, audiopath))
		  '''
        mel = self.get_mel(audio)
        text_encoded = self.get_text(text)
        speaker_id = self.get_speaker_id(speaker_id)
        attn_prior = None
        if self.use_attn_prior:
            attn_prior = self.compute_attention_prior(
                audiopath, mel.shape[1], text_encoded.shape[0])

        return (mel, embeds, speaker_id, text_encoded, attn_prior)

    def __len__(self):
        return len(self.audiopaths_and_text)


class DataCollate():
    """ Zero-pads model inputs and targets based on number of frames per step """
    def __init__(self, n_frames_per_step=1, use_attn_prior=False):
        self.n_frames_per_step = n_frames_per_step
        self.use_attn_prior = use_attn_prior

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[3]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0].item()

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][3]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mel_channels = batch[0][0].size(0)
        max_target_len = max([x[0].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mel_channels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        attn_prior_padded = None
        if self.use_attn_prior:
            attn_prior_padded = torch.FloatTensor(len(batch), max_target_len, max_input_len)
            attn_prior_padded.zero_()
            
        speaker_ids = torch.LongTensor(len(batch))
        #embeds = torch.LongTensor(len(batch))
        embeds = [None for _ in range(len(ids_sorted_decreasing))]
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][0]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
            embeds[i] = batch[ids_sorted_decreasing[i]][1]
            if self.use_attn_prior:
                cur_attn_prior = batch[ids_sorted_decreasing[i]][4]
                attn_prior_padded[i, :cur_attn_prior.size(0), :cur_attn_prior.size(1)] = cur_attn_prior

        return mel_padded, torch.from_numpy(np.array(embeds)), speaker_ids, text_padded, input_lengths, output_lengths, gate_padded, attn_prior_padded


# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-f', '--filelist', type=str,
                        help='List of files to generate mels')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp = Data(**data_config)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    filepaths_and_text = load_filepaths_and_text(args.filelist)
    for (filepath, text, speaker_id) in filepaths_and_text:
        print("speaker id", speaker_id)
        print("text", text)
        print("text encoded", mel2samp.get_text(text))
        audio, sr = load_wav_to_torch(filepath)
        melspectrogram = mel2samp.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)
