import requests

with open('test audio/normal172.mp3', "rb") as audio:
    audio_content = audio.read()

# resp = requests.post("http://127.0.0.1:5000", files={'audio': open('test audio/danger13.mp3', 'rb')})
resp = requests.post(" https://getpredictions-zwofr6ivcq-et.a.run.app/", files={'audio': audio_content})


print(resp.status_code)
if resp.status_code != 204:
      print(resp.json())
else: print(resp.status_code)