from pathlib import Path
import argparse
from multiprocessing.pool import Pool 
from functools import partial
from encoder import inference as encoder
from pathlib import Path
import numpy as np
from tqdm import tqdm
import librosa

def embed_utterance(fpaths, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath = embed_fpath = fpaths
    embed_fpath = embed_fpath.replace(".wav", ".npy")
    wav, rate = librosa.load(wav_fpath)
    wav = encoder.preprocess_wav(wav, rate)
    embed = encoder.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)
    
 
def create_embeddings(filelist_Path: Path, text: str, encoder_model_fpath: Path, n_processes: int):
    metadata_fpath = filelist_Path.joinpath(text)
    assert metadata_fpath.exists()
    
    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [m[0].replace("../","",1) for m in metadata]
        
    func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-f", "--filelist_Path", type=Path, help=\
        "Path to the filelist text file."
        "If you let everything as default, it should be flowtron/filelists/.",
        default="flowtron/filelists/")
    parser.add_argument("-t", "--text", type=str, help= \
        "Text file from filelists.")
    parser.add_argument("-e", "--encoder_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt", help=\
        "Path your trained encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower"
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()
  
    create_embeddings(**vars(args))    
