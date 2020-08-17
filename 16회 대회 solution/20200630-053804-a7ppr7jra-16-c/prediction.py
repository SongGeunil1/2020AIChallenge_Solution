import argparse
import torch
import torchaudio
import logging
import os
from tqdm import tqdm
import soundfile as sf
import numpy as np
import pickle
from tqdm import tqdm, trange
import random


from src.dataloader import data_loader
from src.utils import Vocab
from src.model import CNN2D
import torch.optim as optim
# audio
import librosa
import torchaudio
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.scheduler  import GradualWarmupScheduler

logger = logging.getLogger(__name__)

def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def main():
    model=torch.load('CNNmodel/model.pt')
    model.eval()
    

    config=torch.load('CNNmodel/config.pt')
    
    audio2mfcc = torchaudio.transforms.MFCC(sample_rate=config.sample_rate,
                                        n_mfcc= config.n_mfcc,
                                        log_mels=False,
                                        melkwargs = {'n_fft': config.n_fft_size}).to(config.device)
    
    logger.info('Start cache for test data')
    if not os.path.isfile("./mfcc/test_input.pt"):
        os.makedirs("./mfcc", exist_ok=True)  
        val_root = os.path.join(config.data_path, 'test')
        val_files = [p for p in os.listdir(os.path.join(config.data_path, 'test')) if 'pcm' in p]

        val_mfccs = {}
        for p in tqdm(val_files):
            sound_path = os.path.join(config.data_path, 'test', p)
            data, samplerate  = sf.read(sound_path, channels=1, samplerate=16000,
                          format='raw', subtype='PCM_16')

            mfcc = audio2mfcc(torch.Tensor(data).to(config.device))
            audio_array = torch.zeros(config.n_mfcc, config.input_max_len)
            sel_ix = min(mfcc.shape[1], config.input_max_len)
            audio_array[:,:sel_ix] = mfcc[:,:sel_ix]

            val_mfccs[sound_path] = audio_array.transpose(0,1)
        torch.save(val_mfccs, './mfcc/test_input.pt')
    logger.info('Done cache for test data')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO 

    )


    
    # sentence index
    vocab = Vocab(config)
    test_loader, test_label_path = data_loader( config, 'test', vocab)
    
    with open('prediction.txt', 'w') as file_writer:

        for tst_step, (file_name, mfcc, target_index ) in enumerate(tqdm(test_loader, desc="Evaluating")):
            with torch.no_grad():


                logit, feature= model(mfcc.to(config.device))
                y_max = logit.max(dim=1)[1]
                pred = [vocab.index2text[i] for i in y_max.cpu().numpy()]
                for f, line in zip(file_name, pred): 
                    print(file_name)
                    file_writer.write(f + " "+ str(line)+'\n')

if __name__ == "__main__":
    main()