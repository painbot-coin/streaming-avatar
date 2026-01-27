import uuid
import openai
from flask.sessions import SessionInterface, SessionMixin
from werkzeug.datastructures import CallbackDict
from itertools import cycle, islice
import src.Wav2Lip.audio as audio
from tqdm import tqdm
import numpy as np
from src.Wav2Lip.models import Wav2Lip
from datetime import datetime
import threading
from flask import jsonify, session, stream_with_context
import time
from src.Wav2Lip.inference import video_main
from src.Wav2Lip.inference import datagen
from src.Wav2Lip.inference import face_detect
import torch
from flask import render_template, request, Response
from src import app
import cv2
from text2speech import synthesize_text
import soundfile as sf
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
# from gfpgan import GFPGANer


#Set Initial Variabl
model_path = './src/Wav2Lip/checkpoints/wav2lip.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
face_det_results1 = []
full_frames1 = []
fps = 25
moment = time.time()
srt = time.time()



dni_weight = None
outscale = 1
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
tile = 1000
tile_pad = 10
pre_pad = 10
netscale = 4
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)


# face_enhancer = GFPGANer(
# 			model_path='face_enhance.pth',
# 			upscale=outscale,
# 			arch='clean',
# 			channel_multiplier=2,
# 			bg_upsampler=None)

@app.route('/')
@app.route('/home')
def home():
    ''' Renders the home page '''
    print("!Start")
    return render_template(
        "facesign.html"
    )

def prepare_cha(face_path, ind):
    global full_frames1
    global face_det_results1

    if not os.path.isfile(face_path):
            raise ValueError('--face argument must be a valid path to video/image file')

    elif face_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames1 = [cv2.imread(face_path)]
        fps = 25
    else:
        video_stream = cv2.VideoCapture(face_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            crop = [0, -1, 0, -1]
            y1, y2, x1, x2 = crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames1.append(frame)
    static = False
    box = [-1, -1, -1, -1]
    if box[0] == -1:
        if not static:
            face_det_results1 = face_detect(full_frames1.copy()) # BGR2RGB for CNN face detection
        else:
            face_det_results1 = face_detect([full_frames1.copy()[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = box
        face_det_results1 = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames1.copy()]

prepare_cha('./src/static/video/model.mp4', 1)

def ask_openai(prompt):
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": 'You are the fastest responder.'},{"role": "user", "content": prompt}],
        max_tokens=150
    )  
    answer = response["choices"][0]["message"]["content"]
    return answer

@app.route('/session_start', methods=['GET', 'POST'])
def session_start():
    session["uid"] = str(uuid.uuid1())
    return session["uid"]

def generating_video(id, model):
    res_video = './src/static/video/res{}.mp4'.format(id)
    if os.path.exists(res_video) == True:
        os.remove(res_video)
    video_main(fps, full_frames1, face_det_results1, './audio{}.wav'.format(id), res_video, id, model)

def _load(model_path):
    if device == 'cuda':
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

model = load_model('./src/Wav2Lip/checkpoints/wav2lip.pth')

@app.route('/findanswer', methods=['GET', 'POST'])
def findanswer():
    global wav
    global audioLength
    session['uid'] = str(uuid.uuid1())
    session["question"] = ask_openai(request.args.get('val'))
    session["audio_path"] = './src/static/audio/audio{}.wav'.format(session['uid'])
    global srt
    frt = time.time()
    synthesize_text(session["question"], session["audio_path"])
    f = sf.SoundFile(session["audio_path"])
    audioLength = f.frames / f.samplerate
    print("audioLength = ", session["audio_path"][6:])
    wav = audio.load_wav(session['audio_path'], 16000)
    return jsonify({"audio_path": str("https://avatar.facefile.co/" + session["audio_path"][6:]), "audio_length": audioLength})
    
@app.route('/videostream')
def videostream():
    return Response(stream_with_context(streamvid()), mimetype='multipart/x-mixed-replace; boundary=frame')

def streamvid():
    global full_frames1
    global face_det_results1
    global wav
    global audioLength
    mel = audio.melspectrogram(wav)
    
    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    mel_step_size = 16
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    print("audioLength again = ", audioLength)
    interval = audioLength / len(mel_chunks)
    full_frames = full_frames1
    full_frames = list(islice(cycle(full_frames), len(mel_chunks)))
    face_det_results = face_det_results1
    face_det_results = list(islice(cycle(face_det_results), len(mel_chunks)))
    
    gen = datagen(full_frames.copy(), mel_chunks, face_det_results)
    for j, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(mel_chunks))/256)))):
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
       
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            # _, _, p = face_enhancer.enhance(p, has_aligned=False, only_center_face=True, paste_back=True)
            f[y1:y2, x1:x2] = p
            # f = cv2.resize(f, (1280, 1020))
            _, buffer = cv2.imencode('.jpg', f)
            f = buffer.tobytes()
            stamp = time.time()
            global moment
            duration = stamp - moment
            print(duration, interval)
            if duration < interval:
                time.sleep(interval-duration)
            moment = time.time()
            session['frame'] = f
            yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + session['frame'] + b'\r\n')
