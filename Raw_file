import warnings
warnings.simplefilter('ignore')
from tkinter import *
from tkinter import ttk,filedialog, messagebox
from pydub import AudioSegment
import os
import argparse
import shutil
import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from lib import dataset
from lib import nets
from lib import spec_utils
import itertools
import threading
lst = []
sound = ''
lista1 = []

root = Tk()
root.title('Filtro de Ruído')
p1 = PhotoImage(file = 'models/imagens/sound-waves.png')
 
# Selecionando o Ícone do Tk
root.iconphoto(False, p1)
root.geometry("500x500")

bg = PhotoImage(file = "models/imagens/wp2831915-black-background-png.png") #Escolhendo o imagem de background do Tk
my_label = Label(root, image = bg)
my_label.place(x=0, y=0, relwidth=1, relheight=1)

progress=ttk.Progressbar(root,orient=HORIZONTAL,length=100,mode='indeterminate')
text_box = Text(root, height=8, width=35, padx=15, pady=15, yscrollcommand = True)
text_box.config(state='normal')
text_box = Text(root, height=8, width=35, padx=15, pady=15, yscrollcommand = True)
text_box.tag_configure("center", justify="center")
text_box.tag_add("center", 1.0, "end")
text_box.place(x=100, y=250)
text_box.insert(1.0, 'Aguardando seleção de arquivos...')
text_box.config(state='disabled')

def quit_me(): #Saí do programa de maneira abrupta
    print('\nSaindo...')
    root.destroy()
    
    
def open_file(): #Abre o arquivo
    global lista1, lst
    nomes=[]
    answer = True
    text_box.config(state='normal')
    text_box.delete(1.0,"end")
    text_box.insert(1.0, "Arquivos selecionados:" + "\n")
    if answer == True: #Adiciona a primeira vez
        song = filedialog.askopenfilenames(title="Filtro de ruídos",
                                            filetypes= (("Audio Files",".wav .ogg .mp3 .mpeg .wma .flac .aiff .aac .alac .pcm"),
                                            ("all files","*.*")))
        
    answer = messagebox.askyesno("","Continuar adicionando?")
    
    lst.append(song) 
    while answer == True: #Continua adicionando
        song = filedialog.askopenfilenames(title="Filtro de ruídos",
                                          filetypes= (("Audio Files",".wav .ogg .mp3 .mpeg .wma .flac .aiff .aac .alac .pcm"),
                                          ("all files","*.*")))
        
        answer = messagebox.askyesno("","Continuar adicionando?")
        lst.append(song)
        
    text_box.delete(3.0,"end")                                        
    lista1 = list(itertools.chain(*lst)) #Transforma dicionário em lista
    nomes.clear()
    
    for i in lista1:#Escreve na tela quais arquivos serão filtrados
        new_img = i[:-4]
        img2 = os.path.basename(new_img)
        nomes.append(img2)
        text_box.insert(3.0,"- "+ img2 + "\n")
    text_box.config(state='disabled')
    return lista1
    
    
def saveFile():
    global dirname
    dirname = filedialog.askdirectory(parent=root,initialdir="/",title='Selecione o diretório')
    return dirname

    
def convertFormat(*arg):      #Filtra o ruído
    global dirname, lista_de_audios
    lista_de_audios = []
    
    try: 
        for audio_foradeformato in lista1: #Converte qualquer arquivo para o formato .wav mono 16khz 16bit
            img = os.path.basename(audio_foradeformato)
            if (img.endswith('wav') or img.endswith('aac') or img.endswith('ogg') or img.endswith('wma')  or img.endswith('mp3')  or img.endswith('alac') or img.endswith('flac') or img.endswith('aiff') or img.endswith('pcm')):   
            
                sound = AudioSegment.from_file(audio_foradeformato)
                if (img.endswith('alac') or img.endswith('flac') or img.endswith('aiff')):
                    img_semformato = img[:-5]
                else:
                    img_semformato = img[:-4]
                sound.export(img_semformato+"_CONVERTIDO.wav", format="wav")
                RUIDOSO = img_semformato+"_CONVERTIDO.wav"
                dir = (dirname + "\\" + img_semformato)
                if os.path.exists(dir):
                    messagebox.showwarning(title='Aviso', message= 'Já existe uma pasta com esse nome: ' + img_semformato)
                else:
                    os.makedirs(dir)
                    shutil.copy(audio_foradeformato,dir)
                    lista_de_audios.append(RUIDOSO)
    except NameError:
        messagebox.showwarning(title='Aviso', message= 'Por favor, insira o local de salvamento')
    except FileNotFoundError:
        messagebox.showwarning(title='Aviso', message= 'Arquivo não encontrado') 

    class VocalRemover(object):
        
        def __init__(self, model, device, window_size):
            self.model = model
            self.offset = model.offset
            self.device = device
            self.window_size = window_size
    
        def _execute(self, X_mag_pad, roi_size, n_window):
            self.model.eval()
            with torch.no_grad():
                preds = []
                for i in tqdm(range(n_window)):
                    start = i * roi_size
                    X_mag_window = X_mag_pad[None, :, :, start:start + self.window_size]
                    X_mag_window = torch.from_numpy(X_mag_window).to(self.device)
                    pred = self.model.predict(X_mag_window)
                    pred = pred.detach().cpu().numpy()
                    preds.append(pred[0])
                    
                pred = np.concatenate(preds, axis=2)

            return pred

        def preprocess(self, X_spec):
            X_mag = np.abs(X_spec)
            X_phase = np.angle(X_spec)
    
            return X_mag, X_phase
    
        def inference(self, X_spec):
            X_mag, X_phase = self.preprocess(X_spec)
    
            coef = X_mag.max()
            X_mag_pre = X_mag / coef
    
            n_frame = X_mag_pre.shape[2]
            pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.window_size, self.offset)
            n_window = int(np.ceil(n_frame / roi_size))
    
            X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
    
            pred = self._execute(X_mag_pad, roi_size, n_window)
            pred = pred[:, :, :n_frame]
    
            return pred * coef, X_mag, np.exp(1.j * X_phase)
    
        def inference_tta(self, X_spec):
            X_mag, X_phase = self.preprocess(X_spec)
    
            coef = X_mag.max()
            X_mag_pre = X_mag / coef
    
            n_frame = X_mag_pre.shape[2]
            pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.window_size, self.offset)
            n_window = int(np.ceil(n_frame / roi_size))
    
            X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
    
            pred = self._execute(X_mag_pad, roi_size, n_window)
            pred = pred[:, :, :n_frame]
    
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            n_window += 1
    
            X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
    
            pred_tta = self._execute(X_mag_pad, roi_size, n_window)
            pred_tta = pred_tta[:, :, roi_size // 2:]
            pred_tta = pred_tta[:, :, :n_frame]
    
            return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.j * X_phase)


    def main():
        
        count = 0
        for audio in lista_de_audios:
            count += 1
            progress.start(10)
            p = argparse.ArgumentParser()
            p.add_argument('--gpu', '-g', type=int, default=-1)
            p.add_argument('--pretrained_model', '-P', type=str, default='models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth')
            p.add_argument('--input', '-i', required=False, default = audio)
            p.add_argument('--sr', '-r', type=int, default=44100)
            p.add_argument('--n_fft', '-f', type=int, default=2048)
            p.add_argument('--hop_length', '-l', type=int, default=1024)
            p.add_argument('--window_size', '-w', type=int, default=512)
            p.add_argument('--output_image', '-I', action='store_true')
            p.add_argument('--postprocess', '-p', action='store_true')
            p.add_argument('--tta', '-t', action='store_true')
            args = p.parse_args()
    
          #  print('carregando o modelo...', end=' ')
            device = torch.device('cpu')
            model = nets.CascadedASPPNet(args.n_fft)
            model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
            if torch.cuda.is_available() and args.gpu >= 0:
                device = torch.device('cuda:{}'.format(args.gpu))
                model.to(device)
          #  print('concluído')
    
          #  print('carregando o sinal...', end=' ')
            X, sr = librosa.load(
                args.input, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
            basename = os.path.splitext(os.path.basename(args.input))[0]
          #  print('done')
    
            if X.ndim == 1:
                X = np.asarray([X, X])

          #  print('realizando a filtragem...', end=' ')
            X = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
          #  print('done')
            
            vr = VocalRemover(model, device, args.window_size)
            if args.tta:
                pred, X_mag, X_phase = vr.inference_tta(X)
            else:
                pred, X_mag, X_phase = vr.inference(X)
            

          #  print('separando o áudio do ruído...', end=' ')
            v_spec = np.clip(X_mag - pred, 0, np.inf) * X_phase
            wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
          #  print('concluído')
            sf.write('{}_VOZ.wav'.format(basename), wave.T, sr)
            shutil.move(basename+"_VOZ.wav", dirname + "/" + audio[:-15])
            progress.stop()
            os.remove(audio)
            if count == len(lista_de_audios):
                text_box.config(state='normal')
                text_box.delete(2.0,"end") 
                text_box.config(state='disabled')
                lst.clear()
                lista1.clear()
                lista_de_audios.clear()
                messagebox.showwarning(title='Aviso', message='Tarefa concluída')

           
    if __name__ == '__main__':
        main()


def restart_program(): #Limpa a lista com os áudios
    global lista1,lst
    #text_box.delete(3.0,"end")
    lst.clear()
    lista1.clear()
    text_box.config(state='normal')
    text_box.delete(2.0,"end") 
    text_box.config(state='disabled')
    
browse_text = StringVar()
browse_btn = Button(text = "Carregar Áudio", command= open_file, font="Raleway", bg="gray", fg="white", height=1, width=14)
progress.place(x = 203, y = 411)
browse_btn.place(x=180, y=10)
button3 = Button(text = "Local de salvamento", command = saveFile, font="Raleway", bg="gray", fg="white", height=1, width=17)
button3.place(x=170, y=50)
button3 = Button(text = "Filtrar ruído", command=lambda:threading.Thread(target=convertFormat, daemon = True).start(), font="Raleway", bg="gray", fg="white", height=1, width=17)
button3.place(x=170, y=90)
button4 = Button(text = "Sair", command = quit_me, font="Raleway", bg="gray", fg="white", height=1, width=17)
button4.place(x=335, y=465)
button5 = Button(text = "Limpar dados", command= restart_program, font="Raleway", bg="gray", fg="white", height=1, width=14)
button5.place(x = 3, y = 465)


root.mainloop()
