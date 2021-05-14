# VoiceCloning

A voice cloning application that combines flowtron, waveglow and a GE2E loss encoder that encodes the speaker's voice.

This is an alternative implementation similar with SV2TTS https://github.com/CorentinJ/Real-Time-Voice-Cloning using different components.

Report: https://drive.google.com/file/d/1PY1nduqBCuaJcSActdQs1Ix0WVnvWdjb/view?usp=sharing 

Video Presentation: https://drive.google.com/file/d/1kkUPsXvUgIcIUWBbFZx2yQTpaaigdyqt/view?usp=sharing

#How to run the project

make sure all the required libraries are correctly installed.
python3.7 works the best.

cd flowtron/ 
python3.7 inference.py -c infer.json -w <vocoder path> -e <input audio path> -t <input text> -f <model path>

the results are stored in results/

#Related software

Check out my REST web service for VoiceCloning at https://github.com/VladAIacob/VoiceCloningRESTApi
and my mobile application at https://github.com/VladAIacob/VoiceCloningMobileApp
