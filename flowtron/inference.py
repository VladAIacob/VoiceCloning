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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import os
import argparse
import json
import sys
import numpy as np
import torch
import librosa

sys.path.insert(0, "../")
from encoder import inference as encoder
from pathlib import Path
from flowtron import Flowtron
from torch.utils.data import DataLoader
from data import Data
from train import update_params

sys.path.insert(0, "tacotron2")
sys.path.insert(0, "tacotron2/waveglow")
from glow import WaveGlow
from scipy.io.wavfile import write

def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath = fpaths
    wav, rate = librosa.load(wav_fpath)
    wav = encoder.preprocess_wav(wav, rate)
    return encoder.embed_utterance(wav)

def infer(flowtron_path, waveglow_path, output_dir, text, embeds, n_frames,
          sigma, gate_threshold, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # load waveglow
    waveglow = torch.load(waveglow_path)['model'].cuda().eval()
    waveglow.cuda().half()
    for k in waveglow.convinv:
        k.float()
    waveglow.eval()

    # load flowtron
    model = Flowtron(**model_config).cuda()
    state_dict = torch.load(flowtron_path, map_location='cpu')['model'].state_dict()
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded checkpoint '{}')" .format(flowtron_path))

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))
    text = trainset.get_text(text).cuda()
    embeds = trainset.get_embeds(embeds).cuda()
    text = text[None]
    embeds = embeds[None]

    with torch.no_grad():
        residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
        mels, attentions = model.infer(
            residual, embeds, text, gate_threshold=gate_threshold)

    for k in range(len(attentions)):
        attention = torch.cat(attentions[k]).cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        axes[0].imshow(mels[0].cpu().numpy(), origin='lower', aspect='auto')
        axes[1].imshow(attention[:, 0].transpose(), origin='lower', aspect='auto')
        fig.savefig(os.path.join(output_dir, 'sigma{}_attnlayer{}.png'.format(sigma, k)))
        plt.close("all")

    with torch.no_grad():
        audio = waveglow.infer(mels.half(), sigma=0.8).float()

    audio = audio.cpu().numpy()[0]
    # normalize audio for now
    audio = audio / np.abs(audio).max()
    print(audio.shape)

    write(os.path.join(output_dir, 'sigma{}.wav'.format(sigma)),
          data_config['sampling_rate'], audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-f', '--flowtron_path',
                        help='Path to flowtron state dict', type=str)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow state dict', type=str)
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    parser.add_argument('-e', '--embeds', help='Speaker voice', type=str)
    parser.add_argument('-n', '--n_frames', help='Number of frames',
                        default=400, type=int)
    parser.add_argument('-o', "--output_dir", default="results/")
    parser.add_argument("-s", "--sigma", default=0.5, type=float)
    parser.add_argument("-g", "--gate", default=0.5, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)

    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
   
    embeds = embed_utterance(args.embeds, Path("../encoder/saved_models/pretrained.pt"))
    
    infer(args.flowtron_path, args.waveglow_path, args.output_dir, args.text, [embeds], #args.embeds,
			 args.n_frames, args.sigma, args.gate, args.seed)
