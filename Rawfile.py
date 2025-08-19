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
import queue

class NoiseRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Filtro de Ruído')
        self.root.geometry("500x500")

        self.file_paths = []
        self.output_directory = ''
        self.queue = queue.Queue()

        self.setup_gui()
        self.process_queue()

    def setup_gui(self):
        # Setup background and icons
        script_dir = os.path.dirname(__file__)
        self.p1 = PhotoImage(file = os.path.join(script_dir, 'models', 'imagens', 'sound-waves.png'))
        self.root.iconphoto(False, self.p1)
        self.bg = PhotoImage(file = os.path.join(script_dir, "models", "imagens", "wp2831915-black-background-png.png"))
        my_label = Label(self.root, image=self.bg)
        my_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Setup widgets
        self.progress = ttk.Progressbar(self.root, orient=HORIZONTAL, length=100, mode='indeterminate')
        self.text_box = Text(self.root, height=8, width=35, padx=15, pady=15, yscrollcommand=True)
        self.text_box.tag_configure("center", justify="center")
        self.text_box.tag_add("center", 1.0, "end")
        self.text_box.place(x=100, y=250)
        self.text_box.insert(1.0, 'Aguardando seleção de arquivos...')
        self.text_box.config(state='disabled')

        # Setup buttons
        self.browse_btn = Button(self.root, text="Carregar Áudio", command=self.open_file, font="Raleway", bg="gray", fg="white", height=1, width=14)
        self.browse_btn.place(x=180, y=10)
        self.save_button = Button(self.root, text="Local de salvamento", command=self.select_output_directory, font="Raleway", bg="gray", fg="white", height=1, width=17)
        self.save_button.place(x=170, y=50)
        self.filter_button = Button(self.root, text="Filtrar ruído", command=lambda: threading.Thread(target=self.process_files, daemon=True).start(), font="Raleway", bg="gray", fg="white", height=1, width=17)
        self.filter_button.place(x=170, y=90)
        self.quit_button = Button(self.root, text="Sair", command=self.quit_app, font="Raleway", bg="gray", fg="white", height=1, width=17)
        self.quit_button.place(x=335, y=465)
        self.clear_button = Button(self.root, text="Limpar dados", command=self.clear_selection, font="Raleway", bg="gray", fg="white", height=1, width=14)
        self.clear_button.place(x=3, y=465)
        self.progress.place(x = 203, y = 411)
        
    def open_file(self):
        song_list = []
        answer = True
        while answer:
            songs = filedialog.askopenfilenames(
                title="Filtro de ruídos",
                filetypes=(("Audio Files", ".wav .ogg .mp3 .mpeg .wma .flac .aiff .aac .alac .pcm"),
                           ("all files", "*.*"))
            )
            if songs:
                song_list.extend(songs)
            answer = messagebox.askyesno("", "Continuar adicionando?")

        self.file_paths = song_list
        self.update_textbox()

    def update_textbox(self):
        self.text_box.config(state='normal')
        self.text_box.delete(1.0, "end")
        self.text_box.insert(1.0, "Arquivos selecionados:\n")
        for file_path in self.file_paths:
            base_name = os.path.basename(file_path)
            self.text_box.insert(3.0, f"- {base_name}\n")
        self.text_box.config(state='disabled')

    def select_output_directory(self):
        self.output_directory = filedialog.askdirectory(parent=self.root, initialdir="/", title='Selecione o diretório')

    def process_files(self):
        if not self.file_paths:
            messagebox.showwarning(title='Aviso', message='Nenhum arquivo de áudio selecionado.')
            return
        if not self.output_directory:
            messagebox.showwarning(title='Aviso', message='Por favor, insira o local de salvamento')
            return

        converted_files = []
        for audio_path in self.file_paths:
            try:
                base_name, extension = os.path.splitext(os.path.basename(audio_path))
                if extension.lower() in ['.wav', '.aac', '.ogg', '.wma', '.mp3', '.alac', '.flac', '.aiff', '.pcm']:
                    sound = AudioSegment.from_file(audio_path)
                    converted_filename = f"{base_name}_CONVERTIDO.wav"
                    converted_path = os.path.join(self.output_directory, converted_filename)
                    sound.export(converted_path, format="wav")

                    dir_path = os.path.join(self.output_directory, base_name)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    shutil.copy(audio_path, dir_path)
                    converted_files.append(converted_path)
            except Exception as e:
                messagebox.showerror(title='Erro', message=f'Erro ao converter {os.path.basename(audio_path)}: {e}')

        self.run_vocal_remover(converted_files)

    def run_vocal_remover(self, audio_files):
        # The vocal remover logic will be placed here.
        # For now, it's a placeholder.
        # This will also be refactored to remove argparse and improve clarity.
        
        # Placeholder for the main processing logic
        def main_processing(audio_list, output_dir):
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

            count = 0
            for audio in audio_list:
                count += 1
                self.progress.start(10)

                script_dir = os.path.dirname(__file__)
                model_path = os.path.join(script_dir, 'models', 'MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth')

                sr = 44100
                n_fft = 2048
                hop_length = 1024
                window_size = 512
                tta = False

                device = torch.device('cpu')
                model = nets.CascadedASPPNet(n_fft)
                model.load_state_dict(torch.load(model_path, map_location=device))
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    model.to(device)

                X, sr = librosa.load(audio, sr, False, dtype=np.float32, res_type='kaiser_fast')
                basename = os.path.splitext(os.path.basename(audio))[0]

                if X.ndim == 1:
                    X = np.asarray([X, X])

                X = spec_utils.wave_to_spectrogram(X, hop_length, n_fft)

                vr = VocalRemover(model, device, window_size)
                if tta:
                    pred, X_mag, X_phase = vr.inference_tta(X)
                else:
                    pred, X_mag, X_phase = vr.inference(X)

                v_spec = np.clip(X_mag - pred, 0, np.inf) * X_phase
                wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=hop_length)

                base_name_no_ext = os.path.splitext(os.path.basename(audio))[0].replace('_CONVERTIDO', '')
                output_filename = f"{base_name_no_ext}_VOZ.wav"

                # Corrected output path
                output_path = os.path.join(output_dir, base_name_no_ext, output_filename)
                sf.write(output_path, wave.T, sr)

                self.progress.stop()
                os.remove(audio)
                if count == len(audio_list):
                    self.queue.put(('task_completed', 'Tarefa concluída'))

        try:
            main_processing(audio_files, self.output_directory)
        except Exception as e:
            self.queue.put(('error', f'Ocorreu um erro: {e}'))

    def process_queue(self):
        try:
            message_type, data = self.queue.get_nowait()
            if message_type == 'task_completed':
                messagebox.showinfo(title='Aviso', message=data)
                self.clear_selection()
            elif message_type == 'error':
                messagebox.showerror(title='Erro', message=data)

            self.progress.stop()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def clear_selection(self):
        self.file_paths = []
        self.update_textbox()
        self.text_box.config(state='normal')
        self.text_box.delete(1.0, "end")
        self.text_box.insert(1.0, 'Aguardando seleção de arquivos...')
        self.text_box.config(state='disabled')


    def quit_app(self):
        print('\nSaindo...')
        self.root.destroy()

if __name__ == '__main__':
    root = Tk()
    app = NoiseRemoverApp(root)
    root.mainloop()
