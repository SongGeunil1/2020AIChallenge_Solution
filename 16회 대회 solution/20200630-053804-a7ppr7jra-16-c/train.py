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

def createMFCC(config): 

    '''
    Create MFCC 
    '''
    audio2mfcc = torchaudio.transforms.MFCC(sample_rate=config.sample_rate,
                                        n_mfcc= config.n_mfcc,
                                        log_mels=False,
                                        melkwargs = {'n_fft': config.n_fft_size}).to(config.device)


    logger.info('Start cache for Training data')
    if not os.path.isfile("./mfcc/train_input.pt"):
        os.makedirs("./mfcc", exist_ok=True)  
        train_root = os.path.join(config.data_path, 'train')
        train_files = [p for p in os.listdir(os.path.join(config.data_path, 'train')) if 'pcm' in p]

        trn_mfccs = {}
        for p in tqdm(train_files):
            
            sound_path = os.path.join(config.data_path, 'train', p)
            data, samplerate  = sf.read(sound_path, channels=1, samplerate=16000,
                          format='raw', subtype='PCM_16')

            mfcc = audio2mfcc(torch.Tensor(data).to(config.device))
            audio_array = torch.zeros(config.n_mfcc, config.input_max_len)
            sel_ix = min(mfcc.shape[1], config.input_max_len)
            audio_array[:,:sel_ix] = mfcc[:,:sel_ix]

            trn_mfccs[sound_path] = audio_array.transpose(0,1)
        torch.save(trn_mfccs, './mfcc/train_input.pt')
    logger.info('Done cache for Training data')



    logger.info('Start cache for validate data')
    if not os.path.isfile("./mfcc/validate_input.pt"):
        os.makedirs("./mfcc", exist_ok=True)  
        val_root = os.path.join(config.data_path, 'validate')
        val_files = [p for p in os.listdir(os.path.join(config.data_path, 'validate')) if 'pcm' in p]

        val_mfccs = {}
        for p in tqdm(val_files):
            sound_path = os.path.join(config.data_path, 'validate', p)
            data, samplerate  = sf.read(sound_path, channels=1, samplerate=16000,
                          format='raw', subtype='PCM_16')

            mfcc = audio2mfcc(torch.Tensor(data).to(config.device))
            audio_array = torch.zeros(config.n_mfcc, config.input_max_len)
            sel_ix = min(mfcc.shape[1], config.input_max_len)
            audio_array[:,:sel_ix] = mfcc[:,:sel_ix]

            val_mfccs[sound_path] = audio_array.transpose(0,1)
        torch.save(val_mfccs, './mfcc/validate_input.pt')
    logger.info('Done cache for validate data')
    
    
def evaluate(validate_loader,  loss_fct , model, vocab, config):    
    model.eval()

    val_loss, val_acc, val_f1 = 0, 0, 0

    
    for val_step, (file_name, mfcc, target_index ) in enumerate(tqdm(validate_loader, desc="Evaluating")):
        with torch.no_grad():
            
            audios, texts = list(map(lambda x: x.to(config.device), [mfcc, target_index]))

            texts = texts.squeeze(-1).long().to(config.device)
        

            
            logit, feature= model(audios)

            loss = loss_fct(logit, texts.view(-1))
            
            # max => out: (value, index)
            y_max = logit.max(dim=1)[1]
            
            val_loss += loss.item()
            
    true = [vocab.index2text[i] for i in texts.cpu().numpy()]
    pred = [vocab.index2text[i] for i in y_max.cpu().numpy()]
    pairs = list(zip(true, pred))
    out = '\n'.join([ f'({t}//{p})' for t,p in zip(true, pred)])
    val_loss /= (val_step + 1)
    return val_loss, out


def train(model, train_loader, validate_loader, loss_fct, config, vocab) :   
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # total num_batches
    t_total = len(train_loader) * config.num_epochs
    config.warmup_step = int(config.warmup_percent * t_total)
    
    # decay learning rate, related to a validation
    scheduler_cosine = CosineAnnealingLR(optimizer, config.num_epochs, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(
        optimizer, 1, config.warmup_step, after_scheduler=scheduler_cosine
    )
    
    # Train!
    logger.info(f"***** Running model*****")
    logger.info("  Num Epochs = %d", config.num_epochs)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup Steps = %d", config.warmup_step)
    
    
    
    global_step = 0
    steps_trained_in_current_epoch = 0
    # cum_loss, current loss
    tr_loss, logging_loss = 0.0, 0.0
    best_val_loss = 1e8
    
    
    model.zero_grad()
    
    
    
    train_iterator = trange(
        0, int(config.num_epochs), desc="Epoch",
    )
    
    set_seed(config)
    
    
    
    config.model_path = 'CNNmodel'
    
    if not os.path.isdir(config.model_path):
        os.makedirs(config.model_path)
    
    
    
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_loader, desc="Iteration"
        )
        
        for step, (file_name, mfcc, target_index ) in enumerate(epoch_iterator):
            
            model.train()
            
            audios, texts = list(map(lambda x: x.to(config.device), [mfcc, target_index]))

            texts = texts.squeeze(-1).long().to(config.device)
            
      
            
            logit, feature= model(audios)

            loss = loss_fct(logit, texts.view(-1))
            

            loss.backward()
            
            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            
            optimizer.step()
            scheduler.step()      

            
            model.zero_grad()
            
            global_step += 1

            if global_step % config.logging_steps == 0:
                logger.info("  train loss : %.3f", (tr_loss - logging_loss) / config.logging_steps)
                logging_loss = tr_loss
            

        val_loss, info = evaluate(validate_loader,  loss_fct , model, vocab, config)
        logger.info(info)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model, f"{config.model_path}/model.pt")
            torch.save(config, f"{config.model_path}/config.pt")
            logger.info(f"  Saved {config.model_path}")

        logger.info("  val loss : %.3f", val_loss)
        logger.info("  best_val loss : %.3f", best_val_loss)
    

    
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--input_max_len", default=400, type=int,
                            help="Maximum sequence length for audio")
    args.add_argument("--num_epochs", default=300, type=int,
                            help="num_epochs")
    args.add_argument("--data_path", default='data', type=str,
                            help="root")
    args.add_argument("--sample_rate", default=16000, type=int,
                            help="sampling rate for audio")
    args.add_argument("--n_fft_size", default=400, type=int,
                            help="time widnow for fourier transform")
    args.add_argument("--n_mfcc", default=40, type=int,
                            help="low frequency range (from 0 to n_mfcc)")
    args.add_argument("--max_len", default=30, type=int,
                            help="target_max_length")
    args.add_argument("--batch_size", default=128, type=int,
                            help="target_max_length")
    args.add_argument(
        "--warmup_percent", default=0.1, type=float, help="Linear warmup over warmup_percent."
    )
    args.add_argument(
        "--when", type=int, default=5, help="when to decay learning rate (default: 20)"
    )
    args.add_argument(
        "--clip", type=float, default=0.8, help="gradient clip value (default: 0.8)"
    )
    args.add_argument(
        "--lr", type=float, default=1e-4, help="initial learning rate (default: 1e-3)"
    )
    args.add_argument("--seed", type=int, default=1234, help="random seed")
    args.add_argument(
        "--logging_steps", type=int, default=50, help="frequency of result logging (default: 30)"
    )
    config = args.parse_args()
    set_seed(config)


    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO 

        )


    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device

    
    
    # get mfcc
    createMFCC(config)
    
    # sentence index
    vocab = Vocab(config)
    
    # data loaders
    train_loader, train_label_path = data_loader( config, 'train', vocab)
    validate_loader, validate_label_path = data_loader( config, 'validate', vocab)
    
    # build model(unknown sentence +1)
    model = CNN2D(len(vocab)+1).to(config.device)
    
    # loss function
    loss_fct = torch.nn.CrossEntropyLoss()
    
    train(model, train_loader, validate_loader, loss_fct, config, vocab)
    logger.info('Done Training')
if __name__ =='__main__':
    main()